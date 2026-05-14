"""
Service functions for daily picks and feedback workflows
"""

from typing import Callable
from api.mappers import to_debug_pick, to_public_pick
from api.services.errors import BadRequestError

def ensure_single_category_mvp(get_arxiv_categories: Callable[[], list[str]]) -> None:
    categories = get_arxiv_categories()
    if len(categories) != 1:
        raise BadRequestError("API MVP supports exactly one configured arXiv category")

def get_daily_picks_payload(
    user_id: str,
    profile_id: str | None,
    resolve_profile: Callable[[str, str | None], dict],
    list_digest_selected_profile_ids: Callable[[str], list[str]],
    fetch_latest_picks: Callable[[str], list],
    anchored_run_ids: list[str] | None = None,
) -> dict:
    if profile_id is not None:
        target_profile_ids = [profile_id]
    else:
        target_profile_ids = list_digest_selected_profile_ids(user_id=user_id)
        if not target_profile_ids:
            raise BadRequestError("at least one profile must be selected for digest generation")

    sections = []
    has_anchor_runs = bool(anchored_run_ids)
    for target_profile_id in target_profile_ids:
        profile = resolve_profile(user_id=user_id, profile_id=target_profile_id)
        resolved_profile_id = str(profile["profile_id"])
        rows = fetch_latest_picks(resolved_profile_id)
        # If the caller anchored to specific run IDs, generation already happened for this response.
        section_needs_generation = not rows and not has_anchor_runs
        sections.append(
            {
                "profile_id": resolved_profile_id,
                "profile_slot": profile["profile_slot"],
                "category": profile["category"],
                "interest_sentence": profile["interest_sentence"],
                "needs_generation": section_needs_generation,
                "picks": [to_public_pick(row) for row in rows],
            }
        )

    primary_section = sections[0]

    return {
        "user_id": user_id,
        "profile_id": primary_section["profile_id"],
        "needs_generation": any(section["needs_generation"] for section in sections),
        "picks": primary_section["picks"],
        "sections": sections,
    }

def get_debug_daily_picks_payload(
    user_id: str,
    profile_id: str | None,
    resolve_profile: Callable[[str, str | None], dict],
    fetch_latest_picks: Callable[[str], list],
) -> dict:
    profile = resolve_profile(user_id=user_id, profile_id=profile_id)
    resolved_profile_id = str(profile["profile_id"])
    rows = fetch_latest_picks(resolved_profile_id)

    payload = {
        "user_id": user_id,
        "profile_id": resolved_profile_id,
        "needs_generation": not rows,
        "run_id": None,
        "category": None,
        "generated_at": None,
        "picks": [to_debug_pick(row) for row in rows],
    }

    if rows:
        payload["run_id"] = rows[0].run_id
        payload["category"] = rows[0].category
        payload["generated_at"] = rows[0].generated_at

    return payload

def generate_daily_picks_payload(
    request,
    get_arxiv_categories: Callable[[], list[str]],
    resolve_profile: Callable[[str, str | None], dict],
    list_digest_selected_profile_ids: Callable[[str], list[str]],
    run_pipeline: Callable[..., dict],
    get_daily_picks_payload: Callable[[str, str | None], dict],
) -> dict:
    ensure_single_category_mvp(get_arxiv_categories)

    if request.profile_id is not None:
        profile = resolve_profile(user_id=request.user_id, profile_id=request.profile_id)
        target_profile_ids = [str(profile["profile_id"])]
    else:
        target_profile_ids = list_digest_selected_profile_ids(user_id=request.user_id)
        if not target_profile_ids:
            raise BadRequestError("at least one profile must be selected for digest generation")

    summary = run_pipeline(
        user_id=request.user_id,
        profile_ids=target_profile_ids,
        max_results=request.max_results,
        embedding_limit=request.embedding_limit,
    )
    picks_payload = get_daily_picks_payload(
        user_id=request.user_id,
        profile_id=request.profile_id,
        run_ids=summary["run_ids"],
    )

    recommendations_by_run_profile = summary.get("recommendations_by_run_profile", {})
    recommendation_counts = {
        run_id: sum(len(profile_rows) for profile_rows in profile_map.values())
        for run_id, profile_map in recommendations_by_run_profile.items()
    }
    if not recommendation_counts:
        recommendation_counts = {
            run_id: len(recommendations)
            for run_id, recommendations in summary["recommendations_by_run"].items()
        }

    return {
        "user_id": request.user_id,
        "profile_id": picks_payload["profile_id"],
        "generated_profile_ids": target_profile_ids,
        "run_ids": summary["run_ids"],
        "embedded_count": summary["embedded_count"],
        "recommendation_counts": recommendation_counts,
        "needs_generation": picks_payload["needs_generation"],
        "picks": picks_payload["picks"],
        "sections": picks_payload["sections"],
    }

def save_feedback_payload(
    request,
    resolve_profile: Callable[[str, str | None], dict],
    save_feedback: Callable[..., str],
    update_preference_embedding: Callable[..., None],
) -> dict:
    profile = resolve_profile(user_id=request.user_id, profile_id=request.profile_id)
    resolved_profile_id = str(profile["profile_id"])
    feedback_id = save_feedback(
        arxiv_id=request.arxiv_id,
        label=request.label,
        user_id=request.user_id,
        profile_id=resolved_profile_id,
    )
    update_preference_embedding(
        user_id=request.user_id,
        profile_id=resolved_profile_id,
    )

    return {
        "feedback_id": feedback_id,
        "user_id": request.user_id,
        "profile_id": resolved_profile_id,
        "arxiv_id": request.arxiv_id,
        "label": request.label,
        "preference_updated": True,
    }
