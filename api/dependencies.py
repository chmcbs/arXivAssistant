"""
Helpers to wire dependencies into services
"""

from dataclasses import asdict

import psycopg
from fastapi import HTTPException

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
)
from api.services.daily_picks import (
    get_daily_picks_payload as get_daily_picks_payload_service,
)
from api.services.daily_picks import (
    get_debug_daily_picks_payload as get_debug_daily_picks_payload_service,
)
from api.services.daily_picks import (
    save_feedback_payload as save_feedback_payload_service,
)
from api.services.errors import InternalServerError, NotFoundError
from api.services.metrics import get_metrics_payload as get_metrics_payload_service
from api.services.profiles import (
    add_profile_keyword_payload as add_profile_keyword_payload_service,
)
from api.services.profiles import (
    create_profile_payload as create_profile_payload_service,
)
from api.services.profiles import (
    list_profile_keywords_payload as list_profile_keywords_payload_service,
)
from api.services.profiles import (
    list_profiles_payload as list_profiles_payload_service,
)
from api.services.profiles import (
    remove_profile_keyword_payload as remove_profile_keyword_payload_service,
)
from api.services.profiles import (
    update_digest_selection_payload as update_digest_selection_payload_service,
)
from api.unit_of_work import ApiUnitOfWork, open_api_unit_of_work
from core.config import DEFAULT_USER_ID, get_arxiv_categories
from core.db import get_database_url
from core.preferences import (
    initialize_preference_embedding,
    save_feedback,
    update_preference_embedding,
)
from core.profiles import (
    add_profile_keyword,
    create_profile,
    get_or_create_default_profile,
    list_digest_selected_profile_ids,
    list_profile_keywords,
    remove_profile_keyword,
    set_digest_profile_selection,
)


def _to_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, InternalServerError):
        return HTTPException(status_code=500, detail=str(error))
    if isinstance(error, NotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    return HTTPException(status_code=400, detail=str(error))


def _fetch_profile_by_id(profile_id: str, user_id: str, conn=None):
    return fetch_profile_by_id(
        profile_id=profile_id,
        user_id=user_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
        conn=conn,
    )


def _resolve_profile(user_id: str, profile_id: str | None, conn=None) -> dict:
    profile = resolve_profile(
        user_id=user_id,
        profile_id=profile_id,
        fetch_profile_by_id=lambda profile_id, user_id: _fetch_profile_by_id(
            profile_id=profile_id,
            user_id=user_id,
            conn=conn,
        ),
        get_or_create_default_profile=lambda user_id: get_or_create_default_profile(
            user_id=user_id,
            conn=conn,
        ),
    )
    return profile if isinstance(profile, dict) else asdict(profile)


def _fetch_latest_picks(profile_id: str, run_ids: list[str] | None = None, conn=None):
    return fetch_latest_picks(
        profile_id=profile_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
        run_ids=run_ids,
        conn=conn,
    )


def _fetch_profiles_for_user(user_id: str, conn=None):
    return fetch_profiles_for_user(
        user_id=user_id,
        connect=psycopg.connect,
        database_url=get_database_url(),
        conn=conn,
    )


def _fetch_metrics_rows(latest_runs_limit: int, conn=None):
    return fetch_metrics_rows(
        latest_runs_limit=latest_runs_limit,
        connect=psycopg.connect,
        database_url=get_database_url(),
        conn=conn,
    )


def _run_pipeline(**kwargs):
    from core.pipeline import run_pipeline

    return run_pipeline(**kwargs)


def get_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
    run_ids: list[str] | None = None,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return get_daily_picks_payload_service(
                user_id=user_id,
                profile_id=profile_id,
                anchored_run_ids=run_ids,
                resolve_profile=lambda user_id, profile_id: _resolve_profile(
                    user_id=user_id,
                    profile_id=profile_id,
                    conn=active_uow.conn,
                ),
                list_digest_selected_profile_ids=lambda user_id: list_digest_selected_profile_ids(
                    user_id=user_id,
                    conn=active_uow.conn,
                ),
                fetch_latest_picks=lambda profile_id: _fetch_latest_picks(
                    profile_id=profile_id,
                    run_ids=run_ids,
                    conn=active_uow.conn,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def get_debug_daily_picks_payload(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return get_debug_daily_picks_payload_service(
                user_id=user_id,
                profile_id=profile_id,
                resolve_profile=lambda user_id, profile_id: _resolve_profile(
                    user_id=user_id,
                    profile_id=profile_id,
                    conn=active_uow.conn,
                ),
                fetch_latest_picks=lambda profile_id: _fetch_latest_picks(
                    profile_id=profile_id,
                    conn=active_uow.conn,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def generate_daily_picks_payload(
    request: GenerateDailyPicksRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:

            def run_pipeline_with_uow(**kwargs):
                summary = _run_pipeline(**kwargs)
                active_uow.set_generated_run_ids(summary.get("run_ids", []))
                return summary

            return generate_daily_picks_payload_service(
                request=request,
                get_arxiv_categories=get_arxiv_categories,
                resolve_profile=lambda user_id, profile_id: _resolve_profile(
                    user_id=user_id,
                    profile_id=profile_id,
                    conn=active_uow.conn,
                ),
                list_digest_selected_profile_ids=lambda user_id: list_digest_selected_profile_ids(
                    user_id=user_id,
                    conn=active_uow.conn,
                ),
                run_pipeline=run_pipeline_with_uow,
                get_daily_picks_payload=lambda user_id, profile_id, run_ids=None: get_daily_picks_payload(
                    user_id=user_id,
                    profile_id=profile_id,
                    run_ids=(
                        run_ids if run_ids is not None else active_uow.generated_run_ids
                    ),
                    uow=active_uow,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def save_feedback_payload(
    request: FeedbackRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return save_feedback_payload_service(
                request=request,
                resolve_profile=lambda user_id, profile_id: _resolve_profile(
                    user_id=user_id,
                    profile_id=profile_id,
                    conn=active_uow.conn,
                ),
                save_feedback=lambda **kwargs: save_feedback(
                    conn=active_uow.conn, **kwargs
                ),
                update_preference_embedding=lambda **kwargs: update_preference_embedding(
                    conn=active_uow.conn,
                    **kwargs,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def create_profile_payload(
    request: CreateProfileRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return create_profile_payload_service(
                request=request,
                create_profile=lambda **kwargs: create_profile(
                    conn=active_uow.conn, **kwargs
                ),
                initialize_preference_embedding=lambda **kwargs: initialize_preference_embedding(
                    conn=active_uow.conn,
                    **kwargs,
                ),
                list_profiles_payload=lambda user_id: list_profiles_payload(
                    user_id=user_id,
                    uow=active_uow,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def list_profiles_payload(
    user_id: str = DEFAULT_USER_ID,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return list_profiles_payload_service(
                user_id=user_id,
                fetch_profiles_for_user=lambda user_id: _fetch_profiles_for_user(
                    user_id=user_id,
                    conn=active_uow.conn,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def update_digest_selection_payload(
    request: UpdateDigestSelectionRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return update_digest_selection_payload_service(
                request=request,
                set_digest_profile_selection=lambda **kwargs: set_digest_profile_selection(
                    conn=active_uow.conn,
                    **kwargs,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def add_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return add_profile_keyword_payload_service(
                profile_id=profile_id,
                request=request,
                add_profile_keyword=lambda **kwargs: add_profile_keyword(
                    conn=active_uow.conn,
                    **kwargs,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def remove_profile_keyword_payload(
    profile_id: str,
    request: ManageProfileKeywordRequest,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return remove_profile_keyword_payload_service(
                profile_id=profile_id,
                request=request,
                remove_profile_keyword=lambda **kwargs: remove_profile_keyword(
                    conn=active_uow.conn,
                    **kwargs,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def list_profile_keywords_payload(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return list_profile_keywords_payload_service(
                profile_id=profile_id,
                user_id=user_id,
                list_profile_keywords=lambda **kwargs: list_profile_keywords(
                    conn=active_uow.conn,
                    **kwargs,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error


def get_metrics_payload(
    latest_runs_limit: int = 10,
    uow: ApiUnitOfWork | None = None,
    conn=None,
) -> dict:
    try:
        with open_api_unit_of_work(uow=uow, conn=conn) as active_uow:
            return get_metrics_payload_service(
                latest_runs_limit=latest_runs_limit,
                fetch_metrics_rows=lambda latest_runs_limit: _fetch_metrics_rows(
                    latest_runs_limit=latest_runs_limit,
                    conn=active_uow.conn,
                ),
            )
    except ValueError as error:
        raise _to_http_exception(error) from error
