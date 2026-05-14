"""
Service functions for profile endpoints
"""

from typing import Callable
from api.mappers import to_profile_summary

def create_profile_payload(
    request,
    create_profile: Callable[..., str],
    initialize_preference_embedding: Callable[..., None],
    list_profiles_payload: Callable[[str], dict],
) -> dict:
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

    return {
        "profile": list_profiles_payload(request.user_id)["profiles"][-1]
    }

def list_profiles_payload(
    user_id: str,
    fetch_profiles_for_user: Callable[[str], list],
) -> dict:
    profile_rows = fetch_profiles_for_user(user_id)
    return {
        "user_id": user_id,
        "profiles": [to_profile_summary(row) for row in profile_rows],
    }

def update_digest_selection_payload(request, set_digest_profile_selection: Callable[..., list[str]]) -> dict:
    selected_profile_ids = set_digest_profile_selection(
        profile_ids=request.profile_ids,
        user_id=request.user_id,
    )
    return {
        "user_id": request.user_id,
        "selected_profile_ids": selected_profile_ids,
    }

def add_profile_keyword_payload(
    profile_id: str,
    request,
    add_profile_keyword: Callable[..., list[str]],
) -> dict:
    keywords = add_profile_keyword(
        profile_id=profile_id,
        user_id=request.user_id,
        keyword=request.keyword,
    )
    return {
        "user_id": request.user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }

def remove_profile_keyword_payload(
    profile_id: str,
    request,
    remove_profile_keyword: Callable[..., list[str]],
) -> dict:
    keywords = remove_profile_keyword(
        profile_id=profile_id,
        user_id=request.user_id,
        keyword=request.keyword,
    )
    return {
        "user_id": request.user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }

def list_profile_keywords_payload(
    profile_id: str,
    user_id: str,
    list_profile_keywords: Callable[..., list[str]],
) -> dict:
    keywords = list_profile_keywords(profile_id=profile_id, user_id=user_id)
    return {
        "user_id": user_id,
        "profile_id": profile_id,
        "keywords": keywords,
    }
