"""
Tests the ingestion pipeline
"""

from unittest.mock import Mock
import pytest
import ingestion
from ingestion import clean_id

def test_clean_id_removes_arxiv_version_suffix():
    assert clean_id("2401.12345v2") == "2401.12345"

def test_clean_id_leaves_unversioned_id_unchanged():
    assert clean_id("2401.12345") == "2401.12345"

def test_run_ingestion_completes_run(monkeypatch):
    papers = [object(), object()]

    monkeypatch.setattr(ingestion, "start_run", Mock(return_value="run-123"))
    monkeypatch.setattr(ingestion, "fetch_papers", Mock(return_value=papers))
    monkeypatch.setattr(ingestion, "save_papers", Mock(return_value=2))
    monkeypatch.setattr(ingestion, "complete_run", Mock())
    monkeypatch.setattr(ingestion, "fail_run", Mock())

    run_id = ingestion.run_ingestion("cs.AI", 10)

    assert run_id == "run-123"
    ingestion.start_run.assert_called_once_with("cs.AI", 10)
    ingestion.fetch_papers.assert_called_once_with(category="cs.AI", max_results=10)
    ingestion.save_papers.assert_called_once_with(papers)
    ingestion.complete_run.assert_called_once_with("run-123", 2, 2)
    ingestion.fail_run.assert_not_called()

def test_run_ingestion_marks_run_failed(monkeypatch):
    error = RuntimeError("arxiv failed")

    monkeypatch.setattr(ingestion, "start_run", Mock(return_value="run-123"))
    monkeypatch.setattr(ingestion, "fetch_papers", Mock(side_effect=error))
    monkeypatch.setattr(ingestion, "save_papers", Mock())
    monkeypatch.setattr(ingestion, "complete_run", Mock())
    monkeypatch.setattr(ingestion, "fail_run", Mock())

    with pytest.raises(RuntimeError):
        ingestion.run_ingestion("cs.AI", 10)

    ingestion.fail_run.assert_called_once_with("run-123", "arxiv failed")
    ingestion.save_papers.assert_not_called()
    ingestion.complete_run.assert_not_called()