"""
Helpers to wire dependencies into services
"""

import psycopg
from dataclasses import asdict
from fastapi import HTTPException
from core.config import DEFAULT_USER_ID, get_arxiv_categories
from core.db import get_database_url
from core.preferences import initialize_preference_embedding, save_feedback, update_preference_embedding
from core.profiles import (
    add_profile_keyword,
    create_profile,
    get_or_create_default_profile,
    list_digest_selected_profile_ids,
    list_profile_keywords,
    remove_profile_keyword,
    set_digest_profile_selection,
)
from api.queries.daily_picks import fetch_latest_picks, fetch_profile_by_id
from api.queries.metrics import fetch_metrics_rows
from api.queries.profiles import fetch_profiles_for_user
from api.schemas import (
    CreateProfileRequest,
    FeedbackRequest,
    GenerateDailyPicksRequest,
    ManageProfileKeywordRequest,
    UpdateDigestSelectionRequest,
)
from api.services.common import resolve_profile
from api.services.daily_picks import (
    generate_daily_picks_payload as generate_daily_picks_payload_service,
    get_daily_picks_payload as get_daily_picks_payload_service,
    get_debug_daily_picks_payload as get_debug_daily_picks_payload_service,
    save_feedback_payload as save_feedback_payload_service,
)
from api.services.errors import NotFoundError
from api.services.metrics import get_metrics_payload as get_metrics_payload_service
from api.services.profiles import (
    add_profile_keyword_payload as add_profile_keyword_payload_service,
    create_profile_payload as create_profile_payload_service,
    list_profile_keywords_payload as list_profile_keywords_payload_service,
    list_profiles_payload as list_profiles_payload_service,
    remove_profile_keyword_payload as remove_profile_keyword_payload_service,
    update_digest_selection_payload as update_digest_selection_payload_service,
)

def _to_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, NotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    return HTTPException(status_code=400, detail=str(error))

def _fetch_profile_by_id(profile_id: str, user_id: str):
    return fetch_profile_by_id(
        profile_id=profile_id,
        user_id=user_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
    )

def _resolve_profile(user_id: str, profile_id: str | None) -> dict:
    profile = resolve_profile(
        user_id=user_id,
        profile_id=profile_id,
        fetch_profile_by_id=_fetch_profile_by_id,
        get_or_create_default_profile=get_or_create_default_profile,
    )
    return profile if isinstance(profile, dict) else asdict(profile)

def _fetch_latest_picks(profile_id: str):
    return fetch_latest_picks(
        profile_id=profile_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
    )

def _fetch_profiles_for_user(user_id: str):
    return fetch_profiles_for_user(
        user_id=user_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
    )

def _fetch_metrics_rows(latest_runs_limit: int):
    return fetch_metrics_rows(
        latest_runs_limit=latest_runs_limit,
        connect=psycopg.connect,
        database_url=get_database_url(),
    )

def _run_pipeline(**kwargs):
    from core.pipeline import run_pipeline

    return run_pipeline(**kwargs)

def get_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    try:
        return get_daily_picks_payload_service(
            user_id=user_id,
            profile_id=profile_id,
            resolve_profile=_resolve_profile,
            fetch_latest_picks=_fetch_latest_picks,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def get_debug_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    try:
        return get_debug_daily_picks_payload_service(
            user_id=user_id,
            profile_id=profile_id,
            resolve_profile=_resolve_profile,
            fetch_latest_picks=_fetch_latest_picks,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def generate_daily_picks_payload(request: GenerateDailyPicksRequest) -> dict:
    try:
        return generate_daily_picks_payload_service(
            request=request,
            get_arxiv_categories=get_arxiv_categories,
            resolve_profile=_resolve_profile,
            list_digest_selected_profile_ids=list_digest_selected_profile_ids,
            run_pipeline=_run_pipeline,
            get_daily_picks_payload=get_daily_picks_payload,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def save_feedback_payload(request: FeedbackRequest) -> dict:
    try:
        return save_feedback_payload_service(
            request=request,
            resolve_profile=_resolve_profile,
            save_feedback=save_feedback,
            update_preference_embedding=update_preference_embedding,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def create_profile_payload(request: CreateProfileRequest) -> dict:
    try:
        return create_profile_payload_service(
            request=request,
            create_profile=create_profile,
            initialize_preference_embedding=initialize_preference_embedding,
            list_profiles_payload=list_profiles_payload,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def list_profiles_payload(user_id: str = DEFAULT_USER_ID) -> dict:
    try:
        return list_profiles_payload_service(
            user_id=user_id,
            fetch_profiles_for_user=_fetch_profiles_for_user,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def update_digest_selection_payload(request: UpdateDigestSelectionRequest) -> dict:
    try:
        return update_digest_selection_payload_service(
            request=request,
            set_digest_profile_selection=set_digest_profile_selection,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def add_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    try:
        return add_profile_keyword_payload_service(
            profile_id=profile_id,
            request=request,
            add_profile_keyword=add_profile_keyword,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def remove_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    try:
        return remove_profile_keyword_payload_service(
            profile_id=profile_id,
            request=request,
            remove_profile_keyword=remove_profile_keyword,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def list_profile_keywords_payload(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    try:
        return list_profile_keywords_payload_service(
            profile_id=profile_id,
            user_id=user_id,
            list_profile_keywords=list_profile_keywords,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error

def get_metrics_payload(latest_runs_limit: int = 10) -> dict:
    try:
        return get_metrics_payload_service(
            latest_runs_limit=latest_runs_limit,
            fetch_metrics_rows=_fetch_metrics_rows,
        )
    except ValueError as error:
        raise _to_http_exception(error) from error
