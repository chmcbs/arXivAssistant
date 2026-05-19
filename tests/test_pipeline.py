"""
Tests the full recommendation pipeline
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from core import pipeline
from core.profiles import ProfileRow


def _profile_row(
    profile_id: str,
    *,
    user_id: str = "default",
    category: str = "cs.AI",
) -> ProfileRow:
    return ProfileRow(
        profile_id=profile_id,
        user_id=user_id,
        profile_slot=1,
        profile_name="Profile",
        category=category,
        interest_sentence="test",
        created_at=datetime.now(UTC),
        digest_enabled=True,
    )


def _patch_category_matching(
    monkeypatch,
    *,
    user_id: str = "default",
    profile_categories: dict[str, str] | None = None,
    run_categories: dict[str, str] | None = None,
) -> None:
    profile_categories = profile_categories or {}
    run_categories = run_categories or {}

    def mock_get_profile(profile_id: str):
        category = profile_categories.get(profile_id, "cs.AI")
        return _profile_row(profile_id, user_id=user_id, category=category)

    monkeypatch.setattr(
        pipeline,
        "fetch_run_categories",
        Mock(
            side_effect=lambda run_ids: {
                run_id: run_categories.get(run_id, "cs.AI") for run_id in run_ids
            }
        ),
    )
    monkeypatch.setattr(pipeline, "get_profile", Mock(side_effect=mock_get_profile))


def test_run_pipeline_calls_steps_in_order(monkeypatch):
    calls = []
    _patch_category_matching(monkeypatch)

    monkeypatch.setattr(
        pipeline,
        "run_ingestion",
        Mock(
            side_effect=lambda max_results: calls.append(("run_ingestion", max_results))
            or ["run-1", "run-2"]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_embeddings",
        Mock(side_effect=lambda limit: calls.append(("run_embeddings", limit)) or 5),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_recommendations",
        Mock(
            side_effect=lambda run_id, user_id, profile_id: calls.append(
                ("generate_recommendations", run_id, user_id, profile_id)
            )
            or [{"rank": 1}]
        ),
    )

    summary = pipeline.run_pipeline(
        user_id="default",
        profile_id="profile-1",
        max_results=123,
        embedding_limit=456,
    )

    assert calls == [
        ("run_ingestion", 123),
        ("run_embeddings", 456),
        ("generate_recommendations", "run-1", "default", "profile-1"),
        ("generate_recommendations", "run-2", "default", "profile-1"),
    ]
    assert summary["run_ids"] == ["run-1", "run-2"]
    assert summary["embedded_count"] == 5
    assert summary["recommendations_by_run_profile"] == {
        "run-1": {"profile-1": [{"rank": 1}]},
        "run-2": {"profile-1": [{"rank": 1}]},
    }


def test_run_pipeline_continues_when_recommendation_fails(monkeypatch):
    _patch_category_matching(monkeypatch)
    monkeypatch.setattr(
        pipeline, "run_ingestion", Mock(return_value=["run-1", "run-2"])
    )
    monkeypatch.setattr(pipeline, "run_embeddings", Mock(return_value=3))
    monkeypatch.setattr(
        pipeline,
        "generate_recommendations",
        Mock(side_effect=[RuntimeError("boom"), [{"rank": 1}]]),
    )

    summary = pipeline.run_pipeline(user_id="default", profile_id="profile-1")

    assert summary["recommendations_by_run_profile"] == {
        "run-1": {"profile-1": []},
        "run-2": {"profile-1": [{"rank": 1}]},
    }
    assert summary["recommendation_status_by_run_profile"]["run-1"]["profile-1"]["status"] == "failed"


def test_run_pipeline_generates_for_multiple_profiles(monkeypatch):
    _patch_category_matching(monkeypatch)
    monkeypatch.setattr(pipeline, "run_ingestion", Mock(return_value=["run-1"]))
    monkeypatch.setattr(pipeline, "run_embeddings", Mock(return_value=2))
    monkeypatch.setattr(
        pipeline,
        "generate_recommendations",
        Mock(side_effect=[[{"rank": 1}], [{"rank": 1}, {"rank": 2}]]),
    )

    summary = pipeline.run_pipeline(
        user_id="default",
        profile_ids=["profile-1", "profile-2"],
    )

    assert summary["recommendations_by_run_profile"] == {
        "run-1": {
            "profile-1": [{"rank": 1}],
            "profile-2": [{"rank": 1}, {"rank": 2}],
        }
    }


def test_run_recommendations_for_profiles_skips_non_matching_runs(monkeypatch):
    _patch_category_matching(
        monkeypatch,
        profile_categories={"profile-1": "cs.CL"},
        run_categories={"run-ai": "cs.AI", "run-cl": "cs.CL"},
    )
    generate = Mock(return_value=[{"rank": 1}])
    monkeypatch.setattr(pipeline, "generate_recommendations", generate)

    summary = pipeline.run_recommendations_for_profiles(
        user_id="default",
        profile_ids=["profile-1"],
        run_ids=["run-ai", "run-cl"],
    )

    generate.assert_called_once_with(
        "run-cl",
        user_id="default",
        profile_id="profile-1",
    )
    assert summary["recommendations_by_run_profile"] == {
        "run-cl": {"profile-1": [{"rank": 1}]},
    }


def test_run_shared_pipeline_steps_uses_config_defaults(monkeypatch):
    monkeypatch.setenv("INGESTION_MAX_RESULTS", "111")
    monkeypatch.setenv("EMBEDDING_LIMIT", "222")
    monkeypatch.setattr(pipeline, "run_ingestion", Mock(return_value=["run-1"]))
    monkeypatch.setattr(pipeline, "run_embeddings", Mock(return_value=4))

    summary = pipeline.run_shared_pipeline_steps()

    pipeline.run_ingestion.assert_called_once_with(max_results=111)
    pipeline.run_embeddings.assert_called_once_with(limit=222)
    assert summary == {"run_ids": ["run-1"], "embedded_count": 4}
