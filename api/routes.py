"""
HTTP route definitions for the API service
"""

from fastapi import FastAPI, HTTPException

from api.dependencies import (
    add_profile_keyword_payload,
    create_profile_payload,
    generate_daily_picks_payload,
    get_daily_picks_payload,
    get_debug_daily_picks_payload,
    get_metrics_payload,
    list_profile_keywords_payload,
    list_profiles_payload,
    remove_profile_keyword_payload,
    save_feedback_payload,
    update_digest_selection_payload,
)
from api.schemas import (
    CreateProfileRequest,
    CreateProfileResponse,
    DailyPicksResponse,
    DebugDailyPicksResponse,
    FeedbackRequest,
    FeedbackResponse,
    GenerateDailyPicksRequest,
    GenerateDailyPicksResponse,
    ListProfilesResponse,
    ManageProfileKeywordRequest,
    ManageProfileKeywordResponse,
    UpdateDigestSelectionRequest,
    UpdateDigestSelectionResponse,
)
from core.config import DEFAULT_USER_ID

app = FastAPI(title="arXiv Assistant API")


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


@app.post(
    "/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse
)
def profiles_keywords_add(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    return add_profile_keyword_payload(profile_id=profile_id, request=request)


@app.delete(
    "/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse
)
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
