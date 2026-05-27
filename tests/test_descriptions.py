"""
Tests for LLM description batch helpers
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, Mock

from core.descriptions import (
    MockLLMProvider,
    LLMResult,
    PaperCandidate,
    _build_prompt,
    _process_paper,
    get_llm_provider,
    repeats_title,
    run_description_batch_for_recommendations,
)


@contextmanager
def _fake_scope(connection):
    yield connection


def test_repeats_title_detects_high_overlap():
    title = "Scaling Laws for Neural Language Models"
    description = "Scaling laws for neural language models on large datasets"
    assert repeats_title(title, description) is True


def test_repeats_title_allows_complementary_sentence():
    title = "Scaling Laws for Neural Language Models"
    description = (
        "Empirical analysis shows loss scales predictably with compute, data, and model size."
    )
    assert repeats_title(title, description) is False


def test_build_prompt_includes_retry_note():
    prompt = _build_prompt(
        title="Example Title",
        abstract="Example abstract body.",
        retry=True,
    )
    assert "repeated the title" in prompt
    assert "Example Title" in prompt


def test_get_llm_provider_returns_mock():
    provider = get_llm_provider("mock")
    assert provider.provider_name == "mock"


def test_process_paper_persists_successful_description(monkeypatch):
    paper = PaperCandidate(
        arxiv_id="2601.00001",
        title="A Completely Different Headline About Widgets",
        abstract="We evaluate widget throughput on synthetic workloads.",
        max_score=0.91,
    )
    provider = MockLLMProvider(
        response_text=(
            "Synthetic workload experiments quantify widget throughput limits across hardware tiers."
        )
    )
    persist = Mock(return_value=True)
    monkeypatch.setattr("core.descriptions._persist_description", persist)

    outcome = _process_paper(
        paper,
        provider,
        batch_id="batch-1",
        request_timeout_s=5,
    )

    assert outcome.status == "succeeded"
    persist.assert_called_once()


def test_process_paper_retries_on_title_repetition(monkeypatch):
    paper = PaperCandidate(
        arxiv_id="2601.00002",
        title="Transformers Improve Benchmark Accuracy",
        abstract="We study benchmark accuracy under varied settings.",
        max_score=0.88,
    )
    provider = MockLLMProvider()
    provider.generate = Mock(
        side_effect=[
            LLMResult(
                text="Transformers improve benchmark accuracy on standard tasks.",
                input_tokens=10,
                output_tokens=5,
                latency_ms=1,
            ),
            LLMResult(
                text="Benchmark accuracy gains come from a revised training schedule and data mix.",
                input_tokens=10,
                output_tokens=5,
                latency_ms=1,
            ),
        ]
    )
    persist = Mock(return_value=True)
    monkeypatch.setattr("core.descriptions._persist_description", persist)

    outcome = _process_paper(
        paper,
        provider,
        batch_id="batch-2",
        request_timeout_s=5,
    )

    assert outcome.status == "succeeded"
    assert provider.generate.call_count == 2


def test_run_description_batch_for_recommendations_records_stats(monkeypatch):
    candidates = [
        PaperCandidate(
            arxiv_id="2601.00003",
            title="Different Title About Systems",
            abstract="Abstract text.",
            max_score=0.95,
        )
    ]
    monkeypatch.setattr(
        "core.descriptions.fetch_paper_candidates",
        Mock(return_value=candidates),
    )
    monkeypatch.setattr(
        "core.descriptions._process_paper",
        Mock(
            return_value=Mock(
                arxiv_id="2601.00003",
                status="succeeded",
                input_tokens=10,
                output_tokens=5,
                latency_ms=3,
            )
        ),
    )

    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(
        "core.descriptions.connection_scope",
        lambda conn=None: _fake_scope(connection),
    )

    stats = run_description_batch_for_recommendations(
        run_ids=["run-1"],
        provider=MockLLMProvider(),
    )

    assert stats["candidate_count"] == 1
    assert stats["attempted"] == 1
    assert stats["succeeded"] == 1
    cursor.execute.assert_called()
