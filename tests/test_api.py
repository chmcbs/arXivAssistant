"""
Tests FastAPI service helpers
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from fastapi import HTTPException
import pytest
import api

def _mock_connection_with_cursor(cursor):
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    return connect

def _pick_row(rank=1):
    generated_at = datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc)
    return (
        rank,
        f"2601.0000{rank}",
        f"Paper {rank}",
        f"Abstract {rank}",
        f"https://arxiv.org/pdf/2601.0000{rank}",
        "run-123",
        "cs.AI",
        generated_at,
        0.75,
        0.15,
        0.9,
        "run",
        0,
    )

def test_get_daily_picks_returns_empty_state(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    monkeypatch.setattr(api.psycopg, "connect", _mock_connection_with_cursor(cursor))
    monkeypatch.setattr(
        api,
        "_resolve_profile",
        Mock(return_value={"profile_id": "profile-1"}),
    )

    payload = api.get_daily_picks_payload(user_id="default")

    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "needs_generation": True,
        "picks": [],
    }

def test_get_daily_picks_returns_public_fields(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [_pick_row()]
    monkeypatch.setattr(api.psycopg, "connect", _mock_connection_with_cursor(cursor))
    monkeypatch.setattr(
        api,
        "_resolve_profile",
        Mock(return_value={"profile_id": "profile-1"}),
    )

    payload = api.get_daily_picks_payload(user_id="default")

    assert payload["needs_generation"] is False
    assert payload["picks"] == [
        {
            "rank": 1,
            "arxiv_id": "2601.00001",
            "title": "Paper 1",
            "abstract": "Abstract 1",
            "pdf_url": "https://arxiv.org/pdf/2601.00001",
        }
    ]

def test_get_debug_daily_picks_includes_ranking_metadata(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [_pick_row()]
    monkeypatch.setattr(api.psycopg, "connect", _mock_connection_with_cursor(cursor))
    monkeypatch.setattr(
        api,
        "_resolve_profile",
        Mock(return_value={"profile_id": "profile-1"}),
    )

    payload = api.get_debug_daily_picks_payload(user_id="default")

    assert payload["run_id"] == "run-123"
    assert payload["category"] == "cs.AI"
    assert payload["picks"][0]["final_score"] == 0.9
    assert payload["picks"][0]["base_dense_score"] == 0.75
    assert payload["picks"][0]["keyword_boost"] == 0.15
    assert payload["picks"][0]["candidate_window"] == "run"
    assert payload["picks"][0]["fallback_stage"] == 0

def test_generate_daily_picks_runs_pipeline_and_returns_picks(monkeypatch):
    run_pipeline = Mock(
        return_value={
            "run_ids": ["run-123"],
            "embedded_count": 5,
            "recommendations_by_run": {"run-123": [{"rank": 1}, {"rank": 2}]},
        }
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "pipeline",
        SimpleNamespace(run_pipeline=run_pipeline),
    )
    monkeypatch.setattr(api, "get_arxiv_categories", Mock(return_value=["cs.AI"]))
    monkeypatch.setattr(
        api,
        "_resolve_profile",
        Mock(return_value={"profile_id": "profile-1"}),
    )
    monkeypatch.setattr(
        api,
        "get_daily_picks_payload",
        Mock(
            return_value={
                "user_id": "default",
                "profile_id": "profile-1",
                "needs_generation": False,
                "picks": [{"rank": 1, "arxiv_id": "2601.00001"}],
            }
        ),
    )

    payload = api.generate_daily_picks_payload(
        api.GenerateDailyPicksRequest(
            user_id="default",
            max_results=123,
            embedding_limit=456,
        )
    )

    run_pipeline.assert_called_once_with(
        user_id="default",
        profile_id="profile-1",
        max_results=123,
        embedding_limit=456,
    )
    assert payload["run_ids"] == ["run-123"]
    assert payload["embedded_count"] == 5
    assert payload["recommendation_counts"] == {"run-123": 2}
    assert payload["picks"] == [{"rank": 1, "arxiv_id": "2601.00001"}]

def test_generate_daily_picks_rejects_multiple_categories(monkeypatch):
    monkeypatch.setattr(api, "get_arxiv_categories", Mock(return_value=["cs.AI", "cs.CL"]))

    with pytest.raises(HTTPException) as error:
        api.generate_daily_picks_payload(api.GenerateDailyPicksRequest())

    assert error.value.status_code == 400
    assert "API MVP" in error.value.detail

def test_save_feedback_payload_updates_preferences(monkeypatch):
    monkeypatch.setattr(api, "save_feedback", Mock(return_value="feedback-123"))
    monkeypatch.setattr(api, "update_preference_embedding", Mock())
    monkeypatch.setattr(
        api,
        "_resolve_profile",
        Mock(return_value={"profile_id": "profile-1"}),
    )

    payload = api.save_feedback_payload(
        api.FeedbackRequest(
            user_id="default",
            arxiv_id="2601.00001",
            label="like",
        )
    )

    api.save_feedback.assert_called_once_with(
        arxiv_id="2601.00001",
        label="like",
        user_id="default",
        profile_id="profile-1",
    )
    api.update_preference_embedding.assert_called_once_with(
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

def test_add_profile_keyword_payload_maps_response(monkeypatch):
    monkeypatch.setattr(
        api,
        "add_profile_keyword",
        Mock(return_value=["encoder transformers", "kv cache"]),
    )

    payload = api.add_profile_keyword_payload(
        profile_id="profile-1",
        request=api.ManageProfileKeywordRequest(
            user_id="default",
            keyword="KV Cache",
        ),
    )

    api.add_profile_keyword.assert_called_once_with(
        profile_id="profile-1",
        user_id="default",
        keyword="KV Cache",
    )
    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "keywords": ["encoder transformers", "kv cache"],
    }

def test_remove_profile_keyword_payload_maps_response(monkeypatch):
    monkeypatch.setattr(
        api,
        "remove_profile_keyword",
        Mock(return_value=["encoder transformers"]),
    )

    payload = api.remove_profile_keyword_payload(
        profile_id="profile-1",
        request=api.ManageProfileKeywordRequest(
            user_id="default",
            keyword="KV Cache",
        ),
    )

    api.remove_profile_keyword.assert_called_once_with(
        profile_id="profile-1",
        user_id="default",
        keyword="KV Cache",
    )
    assert payload == {
        "user_id": "default",
        "profile_id": "profile-1",
        "keywords": ["encoder transformers"],
    }

def test_get_metrics_payload_returns_run_and_recommendation_counts(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [("completed", 2), ("failed", 1)],
        [
            (
                "run-123",
                "completed",
                "cs.AI",
                150,
                100,
                100,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                datetime(2026, 1, 2, 1, tzinfo=timezone.utc),
                None,
            )
        ],
        [("profile-1", 3)],
    ]
    cursor.fetchone.return_value = (3,)
    monkeypatch.setattr(api.psycopg, "connect", _mock_connection_with_cursor(cursor))

    payload = api.get_metrics_payload(latest_runs_limit=5)

    assert payload["run_status_counts"] == {"completed": 2, "failed": 1}
    assert payload["latest_runs"][0]["run_id"] == "run-123"
    assert payload["total_recommendations"] == 3
    assert payload["recommendations_by_profile"] == {"profile-1": 3}
