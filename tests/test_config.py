"""
Tests application configuration helpers
"""

import pytest
import config

def test_get_hybrid_weights_uses_defaults(monkeypatch):
    monkeypatch.delenv("HYBRID_DENSE_WEIGHT", raising=False)
    monkeypatch.delenv("HYBRID_KEYWORD_WEIGHT", raising=False)

    assert config.get_hybrid_weights() == (0.6, 0.4)

def test_get_hybrid_weights_reads_environment(monkeypatch):
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "2")
    monkeypatch.setenv("HYBRID_KEYWORD_WEIGHT", "1")

    assert config.get_hybrid_weights() == (2 / 3, 1 / 3)

def test_get_hybrid_weights_rejects_negative_weight(monkeypatch):
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "-1")
    monkeypatch.setenv("HYBRID_KEYWORD_WEIGHT", "1")

    with pytest.raises(ValueError, match="non-negative"):
        config.get_hybrid_weights()

def test_get_hybrid_weights_rejects_zero_total(monkeypatch):
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0")
    monkeypatch.setenv("HYBRID_KEYWORD_WEIGHT", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        config.get_hybrid_weights()

def test_get_arxiv_categories_uses_default(monkeypatch):
    monkeypatch.delenv("ARXIV_CATEGORIES", raising=False)

    assert config.get_arxiv_categories() == ["cs.AI"]

def test_get_arxiv_categories_parses_comma_separated(monkeypatch):
    monkeypatch.setenv("ARXIV_CATEGORIES", "cs.AI, cs.CL, cs.LG")

    assert config.get_arxiv_categories() == ["cs.AI", "cs.CL", "cs.LG"]

def test_get_arxiv_categories_rejects_empty(monkeypatch):
    monkeypatch.setenv("ARXIV_CATEGORIES", "  ,  , ")

    with pytest.raises(ValueError, match="At least one"):
        config.get_arxiv_categories()

def test_get_daily_picks_k_uses_default():
    assert config.get_daily_picks_k() >= 1

def test_get_daily_picks_k_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_DAILY_K", 0)

    with pytest.raises(ValueError, match="must be >= 1"):
        config.get_daily_picks_k()