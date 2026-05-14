"""
Tests FastAPI service helpers
"""

from datetime import datetime, timezone
from unittest.mock import Mock
import pytest
from api.queries.daily_picks import DailyPickRow
from api.queries.metrics import LatestRunRow, MetricsRowSet
from api.schemas import (
    FeedbackRequest,
    GenerateDailyPicksRequest,
    ManageProfileKeywordRequest,
    UpdateDigestSelectionRequest,
)
from api.services.daily_picks import (
    generate_daily_picks_payload,
    get_daily_picks_payload,
    get_debug_daily_picks_payload,
    save_feedback_payload,
)
from api.services.errors import BadRequestError
from api.services.metrics import get_metrics_payload
from api.services.profiles import (
    add_profile_keyword_payload,
    remove_profile_keyword_payload,
    update_digest_selection_payload,
)

def _pick_row(rank=1):
    return DailyPickRow(
        rank=rank,
        arxiv_id=f"2601.0000{rank}",
        title=f"Paper {rank}",
        abstract=f"Abstract {rank}",
        pdf_url=f"https://arxiv.org/pdf/2601.0000{rank}",
        run_id="run-123",
        category="cs.AI",
        generated_at=datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc),
        base_dense_score=0.75,
        keyword_boost=0.15,
        final_score=0.9,
        candidate_window="run",
        fallback_stage=0,
    )

def test_get_daily_picks_returns_empty_state():
    payload = get_daily_picks_payload(
        user_id="default",
        profile_id=None,
        resolve_profile=Mock(return_value={"profile_id": "profile-1"}),
        fetch_latest_picks=Mock(return_value=[]),
    )

    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "needs_generation": True,
        "picks": [],
    }

def test_get_daily_picks_returns_public_fields():
    payload = get_daily_picks_payload(
        user_id="default",
        profile_id=None,
        resolve_profile=Mock(return_value={"profile_id": "profile-1"}),
        fetch_latest_picks=Mock(return_value=[_pick_row()]),
    )

    assert payload["needs_generation"] is False
    assert payload["picks"] == [
        {
            "rank": 1,
            "arxiv_id": "2601.00001",
            "title": "Paper 1",
            "abstract": "Abstract 1",
            "pdf_url": "https://arxiv.org/pdf/2601.00001",
            "final_score": 0.9,
        }
    ]

def test_get_debug_daily_picks_includes_ranking_metadata():
    payload = get_debug_daily_picks_payload(
        user_id="default",
        profile_id=None,
        resolve_profile=Mock(return_value={"profile_id": "profile-1"}),
        fetch_latest_picks=Mock(return_value=[_pick_row()]),
    )

    assert payload["run_id"] == "run-123"
    assert payload["category"] == "cs.AI"
    assert payload["picks"][0]["final_score"] == 0.9
    assert payload["picks"][0]["base_dense_score"] == 0.75
    assert payload["picks"][0]["keyword_boost"] == 0.15
    assert payload["picks"][0]["candidate_window"] == "run"
    assert payload["picks"][0]["fallback_stage"] == 0

def test_generate_daily_picks_runs_pipeline_and_returns_picks():
    run_pipeline = Mock(
        return_value={
            "run_ids": ["run-123"],
            "embedded_count": 5,
            "recommendations_by_run": {"run-123": [{"rank": 1}, {"rank": 2}]},
        }
    )

    payload = generate_daily_picks_payload(
        GenerateDailyPicksRequest(
            user_id="default",
            max_results=123,
            embedding_limit=456,
        ),
        get_arxiv_categories=Mock(return_value=["cs.AI"]),
        resolve_profile=Mock(),
        list_digest_selected_profile_ids=Mock(return_value=["profile-1", "profile-2"]),
        run_pipeline=run_pipeline,
        get_daily_picks_payload=Mock(
            return_value={
                "user_id": "default",
                "profile_id": "profile-1",
                "needs_generation": False,
                "picks": [{"rank": 1, "arxiv_id": "2601.00001"}],
            }
        ),
    )

    run_pipeline.assert_called_once_with(
        user_id="default",
        profile_ids=["profile-1", "profile-2"],
        max_results=123,
        embedding_limit=456,
    )
    assert payload["generated_profile_ids"] == ["profile-1", "profile-2"]
    assert payload["run_ids"] == ["run-123"]
    assert payload["embedded_count"] == 5
    assert payload["recommendation_counts"] == {"run-123": 2}
    assert payload["picks"] == [{"rank": 1, "arxiv_id": "2601.00001"}]

def test_generate_daily_picks_rejects_multiple_categories():
    with pytest.raises(BadRequestError) as error:
        generate_daily_picks_payload(
            GenerateDailyPicksRequest(),
            get_arxiv_categories=Mock(return_value=["cs.AI", "cs.CL"]),
            resolve_profile=Mock(),
            list_digest_selected_profile_ids=Mock(return_value=["profile-1"]),
            run_pipeline=Mock(),
            get_daily_picks_payload=Mock(),
        )

    assert "API MVP" in str(error.value)

def test_generate_daily_picks_rejects_when_no_digest_profiles_selected():
    with pytest.raises(BadRequestError) as error:
        generate_daily_picks_payload(
            GenerateDailyPicksRequest(user_id="default"),
            get_arxiv_categories=Mock(return_value=["cs.AI"]),
            resolve_profile=Mock(),
            list_digest_selected_profile_ids=Mock(return_value=[]),
            run_pipeline=Mock(),
            get_daily_picks_payload=Mock(),
        )

    assert "at least one profile" in str(error.value)

def test_save_feedback_payload_updates_preferences():
    save_feedback = Mock(return_value="feedback-123")
    update_preference_embedding = Mock()
    payload = save_feedback_payload(
        FeedbackRequest(
            user_id="default",
            arxiv_id="2601.00001",
            label="like",
        ),
        resolve_profile=Mock(return_value={"profile_id": "profile-1"}),
        save_feedback=save_feedback,
        update_preference_embedding=update_preference_embedding,
    )

    save_feedback.assert_called_once_with(
        arxiv_id="2601.00001",
        label="like",
        user_id="default",
        profile_id="profile-1",
    )
    update_preference_embedding.assert_called_once_with(
        user_id="default",
        profile_id="profile-1",
    )
    assert payload == {
        "feedback_id": "feedback-123",
        "user_id": "default",
        "profile_id": "profile-1",
        "arxiv_id": "2601.00001",
        "label": "like",
        "preference_updated": True,
    }

def test_add_profile_keyword_payload_maps_response():
    add_profile_keyword = Mock(return_value=["encoder transformers", "kv cache"])
    payload = add_profile_keyword_payload(
        profile_id="profile-1",
        request=ManageProfileKeywordRequest(
            user_id="default",
            keyword="KV Cache",
        ),
        add_profile_keyword=add_profile_keyword,
    )

    add_profile_keyword.assert_called_once_with(
        profile_id="profile-1",
        user_id="default",
        keyword="KV Cache",
    )
    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "keywords": ["encoder transformers", "kv cache"],
    }

def test_remove_profile_keyword_payload_maps_response():
    remove_profile_keyword = Mock(return_value=["encoder transformers"])
    payload = remove_profile_keyword_payload(
        profile_id="profile-1",
        request=ManageProfileKeywordRequest(
            user_id="default",
            keyword="KV Cache",
        ),
        remove_profile_keyword=remove_profile_keyword,
    )

    remove_profile_keyword.assert_called_once_with(
        profile_id="profile-1",
        user_id="default",
        keyword="KV Cache",
    )
    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "keywords": ["encoder transformers"],
    }

def test_update_digest_selection_payload_maps_response():
    set_digest_profile_selection = Mock(return_value=["profile-2", "profile-3"])
    payload = update_digest_selection_payload(
        UpdateDigestSelectionRequest(
            user_id="default",
            profile_ids=["profile-2", "profile-3"],
        ),
        set_digest_profile_selection=set_digest_profile_selection,
    )

    set_digest_profile_selection.assert_called_once_with(
        profile_ids=["profile-2", "profile-3"],
        user_id="default",
    )
    assert payload == {
        "user_id": "default",
        "selected_profile_ids": ["profile-2", "profile-3"],
    }

def test_get_metrics_payload_returns_run_and_recommendation_counts():
    metrics_rows = MetricsRowSet(
        run_status_counts={"completed": 2, "failed": 1},
        latest_runs=[
            LatestRunRow(
                run_id="run-123",
                status="completed",
                category="cs.AI",
                max_results=150,
                fetched_count=100,
                saved_count=100,
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                finished_at=datetime(2026, 1, 2, 1, tzinfo=timezone.utc),
                error_message=None,
            )
        ],
        total_recommendations=3,
        recommendations_by_profile={"profile-1": 3},
    )
    payload = get_metrics_payload(
        latest_runs_limit=5,
        fetch_metrics_rows=Mock(return_value=metrics_rows),
    )

    assert payload["run_status_counts"] == {"completed": 2, "failed": 1}
    assert payload["latest_runs"][0]["run_id"] == "run-123"
    assert payload["total_recommendations"] == 3
    assert payload["recommendations_by_profile"] == {"profile-1": 3}
