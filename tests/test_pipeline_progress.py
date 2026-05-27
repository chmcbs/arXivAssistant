"""
Tests pipeline progress monitoring
"""

from core.pipeline_progress import (
    STEP_LABELS,
    clear,
    get_progress,
    set_step,
    track_pipeline,
)


def test_get_progress_inactive_when_not_tracking():
    clear("user-a")
    snapshot = get_progress("user-a")
    assert snapshot.active is False
    assert snapshot.step is None


def test_track_pipeline_reports_steps_and_clears():
    clear("user-b")
    with track_pipeline("user-b"):
        set_step("ingestion")
        mid = get_progress("user-b")
        assert mid.active is True
        assert mid.step == "ingestion"
        assert mid.label == STEP_LABELS["ingestion"]
        set_step("embeddings", detail="Embedded 3 paper(s)")
        detailed = get_progress("user-b")
        assert detailed.detail == "Embedded 3 paper(s)"

    after = get_progress("user-b")
    assert after.active is False
