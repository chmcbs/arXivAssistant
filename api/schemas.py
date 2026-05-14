"""
Pydantic models used by API routes and helper payloads
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID, get_arxiv_categories


class PublicPick(BaseModel):
    rank: int
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str | None
    final_score: float


class DebugPick(PublicPick):
    run_id: str
    category: str
    generated_at: datetime
    base_dense_score: float
    keyword_boost: float
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


class DigestSection(BaseModel):
    profile_id: str
    profile_slot: int
    category: str
    interest_sentence: str
    needs_generation: bool
    picks: list[PublicPick]


class DailyPicksResponse(BaseModel):
    user_id: str
    profile_id: str
    needs_generation: bool
    picks: list[PublicPick]
    sections: list[DigestSection]


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


class GenerationProfileStatus(BaseModel):
    profile_id: str
    status: Literal["succeeded", "failed"]
    recommendation_count: int = Field(ge=0)
    error_message: str | None = None


class GenerationRunStatus(BaseModel):
    run_id: str
    profile_statuses: list[GenerationProfileStatus]


class GenerateDailyPicksResponse(BaseModel):
    user_id: str
    primary_profile_id: str
    requested_profile_ids: list[str]
    run_ids: list[str]
    embedded_count: int
    generation_runs: list[GenerationRunStatus]
    has_failures: bool
    needs_generation: bool
    picks: list[PublicPick]
    sections: list[DigestSection]


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
