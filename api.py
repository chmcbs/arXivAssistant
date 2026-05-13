"""
FastAPI service for personalised arXiv paper recommendations
"""

from datetime import datetime
from typing import Literal
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID, get_arxiv_categories
from db_helper import get_database_url
from preferences import initialize_preference_embedding, save_feedback, update_preference_embedding
from profiles import (
    add_profile_keyword,
    create_profile,
    get_or_create_default_profile,
    list_digest_selected_profile_ids,
    list_profile_keywords,
    remove_profile_keyword,
    set_digest_profile_selection,
)

app = FastAPI(title="arXiv Assistant API")

LATEST_DAILY_PICKS_SQL = """
WITH latest_run AS (
    SELECT
        run_id,
        MAX(generated_at) AS generated_at
    FROM recommendations
    WHERE profile_id = %s
    GROUP BY run_id
    ORDER BY MAX(generated_at) DESC
    LIMIT 1
)
SELECT
    rec.rank,
    p.arxiv_id,
    p.title,
    COALESCE(p.abstract, '') AS abstract,
    p.pdf_url,
    rec.run_id::text,
    r.category,
    rec.generated_at,
    rec.base_dense_score,
    rec.keyword_boost,
    rec.final_score,
    rec.candidate_window,
    rec.fallback_stage
FROM latest_run lr
JOIN recommendations rec
  ON rec.run_id = lr.run_id
 AND rec.profile_id = %s
JOIN papers p ON p.arxiv_id = rec.arxiv_id
JOIN runs r ON r.run_id = rec.run_id
ORDER BY rec.rank ASC;
"""

RUN_STATUS_COUNTS_SQL = """
SELECT status, COUNT(*)
FROM runs
GROUP BY status
ORDER BY status ASC;
"""

LATEST_RUNS_SQL = """
SELECT
    run_id::text,
    status,
    category,
    max_results,
    fetched_count,
    saved_count,
    started_at,
    finished_at,
    error_message
FROM runs
ORDER BY started_at DESC
LIMIT %s;
"""

RECOMMENDATION_TOTAL_SQL = """
SELECT COUNT(*)
FROM recommendations;
"""

RECOMMENDATIONS_BY_PROFILE_SQL = """
SELECT profile_id::text, COUNT(*)
FROM recommendations
GROUP BY profile_id
ORDER BY profile_id ASC;
"""

PROFILE_LIST_SQL = """
SELECT
    p.profile_id::text,
    p.user_id,
    p.profile_slot,
    p.category,
    p.interest_sentence,
    p.digest_enabled,
    p.created_at,
    pp.updated_at,
    COALESCE(
        ARRAY(
            SELECT pk.keyword
            FROM profile_keywords pk
            WHERE pk.profile_id = p.profile_id
            ORDER BY pk.keyword ASC
        ),
        ARRAY[]::text[]
    ) AS keywords
FROM user_profiles p
LEFT JOIN profile_preferences pp ON pp.profile_id = p.profile_id
WHERE p.user_id = %s
ORDER BY p.profile_slot ASC;
"""

class PublicPick(BaseModel):
    rank: int
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str | None

class DebugPick(PublicPick):
    run_id: str
    category: str
    generated_at: datetime
    base_dense_score: float
    keyword_boost: float
    final_score: float
    candidate_window: str
    fallback_stage: int

class ProfileSummary(BaseModel):
    profile_id: str
    user_id: str
    profile_slot: int
    category: str
    interest_sentence: str
    digest_enabled: bool
    keywords: list[str]
    created_at: datetime
    preference_updated_at: datetime | None = None

class DailyPicksResponse(BaseModel):
    user_id: str
    profile_id: str
    needs_generation: bool
    picks: list[PublicPick]

class DebugDailyPicksResponse(BaseModel):
    user_id: str
    profile_id: str
    needs_generation: bool
    run_id: str | None = None
    category: str | None = None
    generated_at: datetime | None = None
    picks: list[DebugPick]

class GenerateDailyPicksRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    profile_id: str | None = None
    max_results: int = Field(default=150, ge=1)
    embedding_limit: int = Field(default=600, ge=1)

class GenerateDailyPicksResponse(BaseModel):
    user_id: str
    profile_id: str
    generated_profile_ids: list[str]
    run_ids: list[str]
    embedded_count: int
    recommendation_counts: dict[str, int]
    needs_generation: bool
    picks: list[PublicPick]

class FeedbackRequest(BaseModel):
    arxiv_id: str
    label: Literal["like", "dislike"]
    user_id: str = DEFAULT_USER_ID
    profile_id: str | None = None

class FeedbackResponse(BaseModel):
    feedback_id: str
    user_id: str
    profile_id: str
    arxiv_id: str
    label: Literal["like", "dislike"]
    preference_updated: bool

class CreateProfileRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    category: str = Field(default_factory=lambda: get_arxiv_categories()[0])
    interest_sentence: str = DEFAULT_INTEREST_TEXT

class CreateProfileResponse(BaseModel):
    profile: ProfileSummary

class ListProfilesResponse(BaseModel):
    user_id: str
    profiles: list[ProfileSummary]

class ManageProfileKeywordRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    keyword: str

class ManageProfileKeywordResponse(BaseModel):
    user_id: str
    profile_id: str
    keywords: list[str]

class UpdateDigestSelectionRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    profile_ids: list[str]

class UpdateDigestSelectionResponse(BaseModel):
    user_id: str
    selected_profile_ids: list[str]

def _resolve_profile(user_id: str, profile_id: str | None) -> dict:
    if profile_id:
        with psycopg.connect(get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT profile_id::text, user_id, profile_slot, category, interest_sentence, created_at
                    FROM user_profiles
                    WHERE profile_id = %s
                      AND user_id = %s;
                    """,
                    (profile_id, user_id),
                )
                row = cur.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="profile not found for user")

        return {
            "profile_id": row[0],
            "user_id": row[1],
            "profile_slot": int(row[2]),
            "category": row[3],
            "interest_sentence": row[4],
            "created_at": row[5],
        }

    return get_or_create_default_profile(user_id=user_id)

def _fetch_latest_picks(profile_id: str) -> list[tuple]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(LATEST_DAILY_PICKS_SQL, (profile_id, profile_id))
            return cur.fetchall()

def _public_pick(row: tuple) -> dict:
    return {
        "rank": int(row[0]),
        "arxiv_id": row[1],
        "title": row[2],
        "abstract": row[3],
        "pdf_url": row[4],
    }

def _debug_pick(row: tuple) -> dict:
    return {
        **_public_pick(row),
        "run_id": row[5],
        "category": row[6],
        "generated_at": row[7],
        "base_dense_score": float(row[8]),
        "keyword_boost": float(row[9]),
        "final_score": float(row[10]),
        "candidate_window": row[11],
        "fallback_stage": int(row[12]),
    }

def get_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    profile = _resolve_profile(user_id=user_id, profile_id=profile_id)
    resolved_profile_id = str(profile["profile_id"])
    rows = _fetch_latest_picks(resolved_profile_id)

    return {
        "user_id": user_id,
        "profile_id": resolved_profile_id,
        "needs_generation": not rows,
        "picks": [_public_pick(row) for row in rows],
    }

def get_debug_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    profile = _resolve_profile(user_id=user_id, profile_id=profile_id)
    resolved_profile_id = str(profile["profile_id"])
    rows = _fetch_latest_picks(resolved_profile_id)

    payload = {
        "user_id": user_id,
        "profile_id": resolved_profile_id,
        "needs_generation": not rows,
        "run_id": None,
        "category": None,
        "generated_at": None,
        "picks": [_debug_pick(row) for row in rows],
    }

    if rows:
        payload["run_id"] = rows[0][5]
        payload["category"] = rows[0][6]
        payload["generated_at"] = rows[0][7]

    return payload

def _ensure_single_category_mvp() -> None:
    categories = get_arxiv_categories()
    if len(categories) != 1:
        raise HTTPException(
            status_code=400,
            detail="API MVP supports exactly one configured arXiv category",
        )

def generate_daily_picks_payload(request: GenerateDailyPicksRequest) -> dict:
    _ensure_single_category_mvp()
    if request.profile_id is not None:
        profile = _resolve_profile(user_id=request.user_id, profile_id=request.profile_id)
        target_profile_ids = [str(profile["profile_id"])]
    else:
        target_profile_ids = list_digest_selected_profile_ids(user_id=request.user_id)
        if not target_profile_ids:
            raise HTTPException(
                status_code=400,
                detail="at least one profile must be selected for digest generation",
            )

    primary_profile_id = target_profile_ids[0]

    from pipeline import run_pipeline

    summary = run_pipeline(
        user_id=request.user_id,
        profile_ids=target_profile_ids,
        max_results=request.max_results,
        embedding_limit=request.embedding_limit,
    )
    picks_payload = get_daily_picks_payload(
        user_id=request.user_id,
        profile_id=primary_profile_id,
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
        "profile_id": primary_profile_id,
        "generated_profile_ids": target_profile_ids,
        "run_ids": summary["run_ids"],
        "embedded_count": summary["embedded_count"],
        "recommendation_counts": recommendation_counts,
        "needs_generation": picks_payload["needs_generation"],
        "picks": picks_payload["picks"],
    }

def save_feedback_payload(request: FeedbackRequest) -> dict:
    try:
        profile = _resolve_profile(user_id=request.user_id, profile_id=request.profile_id)
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
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "feedback_id": feedback_id,
        "user_id": request.user_id,
        "profile_id": resolved_profile_id,
        "arxiv_id": request.arxiv_id,
        "label": request.label,
        "preference_updated": True,
    }

def create_profile_payload(request: CreateProfileRequest) -> dict:
    try:
        profile_id = create_profile(
            user_id=request.user_id,
            category=request.category,
            interest_sentence=request.interest_sentence,
        )
        initialize_preference_embedding(
            interest_text=request.interest_sentence,
            user_id=request.user_id,
            profile_id=profile_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "profile": list_profiles_payload(request.user_id)["profiles"][-1]
    }

def list_profiles_payload(user_id: str = DEFAULT_USER_ID) -> dict:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(PROFILE_LIST_SQL, (user_id,))
            rows = cur.fetchall()

    return {
        "user_id": user_id,
        "profiles": [
            {
                "profile_id": row[0],
                "user_id": row[1],
                "profile_slot": int(row[2]),
                "category": row[3],
                "interest_sentence": row[4],
                "digest_enabled": bool(row[5]),
                "created_at": row[6],
                "preference_updated_at": row[7],
                "keywords": list(row[8] or []),
            }
            for row in rows
        ],
    }

def update_digest_selection_payload(request: UpdateDigestSelectionRequest) -> dict:
    try:
        selected_profile_ids = set_digest_profile_selection(
            profile_ids=request.profile_ids,
            user_id=request.user_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "user_id": request.user_id,
        "selected_profile_ids": selected_profile_ids,
    }

def add_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    try:
        keywords = add_profile_keyword(
            profile_id=profile_id,
            user_id=request.user_id,
            keyword=request.keyword,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "user_id": request.user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }

def remove_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    try:
        keywords = remove_profile_keyword(
            profile_id=profile_id,
            user_id=request.user_id,
            keyword=request.keyword,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "user_id": request.user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }

def list_profile_keywords_payload(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    try:
        keywords = list_profile_keywords(profile_id=profile_id, user_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "user_id": user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }

def get_metrics_payload(latest_runs_limit: int = 10) -> dict:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(RUN_STATUS_COUNTS_SQL)
            run_status_counts = {
                status: int(count)
                for status, count in cur.fetchall()
            }

            cur.execute(LATEST_RUNS_SQL, (latest_runs_limit,))
            latest_runs = [
                {
                    "run_id": row[0],
                    "status": row[1],
                    "category": row[2],
                    "max_results": int(row[3]),
                    "fetched_count": int(row[4] or 0),
                    "saved_count": int(row[5] or 0),
                    "started_at": row[6],
                    "finished_at": row[7],
                    "error_message": row[8],
                }
                for row in cur.fetchall()
            ]

            cur.execute(RECOMMENDATION_TOTAL_SQL)
            total_recommendations = int(cur.fetchone()[0])

            cur.execute(RECOMMENDATIONS_BY_PROFILE_SQL)
            recommendations_by_profile = {
                profile_id: int(count)
                for profile_id, count in cur.fetchall()
            }

    return {
        "run_status_counts": run_status_counts,
        "latest_runs": latest_runs,
        "total_recommendations": total_recommendations,
        "recommendations_by_profile": recommendations_by_profile,
    }

@app.get("/daily-picks", response_model=DailyPicksResponse)
def daily_picks(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    return get_daily_picks_payload(user_id=user_id, profile_id=profile_id)

@app.get("/daily-picks/debug", response_model=DebugDailyPicksResponse)
def daily_picks_debug(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    return get_debug_daily_picks_payload(user_id=user_id, profile_id=profile_id)

@app.post("/daily-picks/generate", response_model=GenerateDailyPicksResponse)
def daily_picks_generate(request: GenerateDailyPicksRequest) -> dict:
    return generate_daily_picks_payload(request)

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> dict:
    return save_feedback_payload(request)

@app.post("/profiles", response_model=CreateProfileResponse)
def profiles_create(request: CreateProfileRequest) -> dict:
    return create_profile_payload(request)

@app.get("/profiles", response_model=ListProfilesResponse)
def profiles_list(user_id: str = DEFAULT_USER_ID) -> dict:
    return list_profiles_payload(user_id=user_id)

@app.get("/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse)
def profiles_keywords_list(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    return list_profile_keywords_payload(profile_id=profile_id, user_id=user_id)

@app.post("/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse)
def profiles_keywords_add(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    return add_profile_keyword_payload(profile_id=profile_id, request=request)

@app.delete("/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse)
def profiles_keywords_remove(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    return remove_profile_keyword_payload(profile_id=profile_id, request=request)

@app.put("/profiles/digest-selection", response_model=UpdateDigestSelectionResponse)
def profiles_digest_selection_update(request: UpdateDigestSelectionRequest) -> dict:
    return update_digest_selection_payload(request)

@app.get("/metrics")
def metrics(latest_runs_limit: int = 10) -> dict:
    if latest_runs_limit < 1:
        raise HTTPException(status_code=400, detail="latest_runs_limit must be >= 1")

    return get_metrics_payload(latest_runs_limit=latest_runs_limit)
