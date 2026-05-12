"""
Tests the retrieval pipeline
"""

from unittest.mock import MagicMock, Mock
import retrieval
from retrieval import clean_query
import pytest

def test_clean_query_strips_whitespace():
    assert clean_query("  recursive agents  ") == "recursive agents"

########################################
############## KEYWORD #################
########################################

def test_search_keyword_papers_returns_empty_for_blank_query():
    assert retrieval.search_keyword_papers("   ") == []

def test_search_keyword_papers_maps_database_rows(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("2604.25917", "Recursive Multi-Agent Systems", None, 3.8),
    ]

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(retrieval.psycopg, "connect", connect)

    results = retrieval.search_keyword_papers("recursive", limit=5)

    assert results == [
        {
            "arxiv_id": "2604.25917",
            "title": "Recursive Multi-Agent Systems",
            "abstract": "",
            "keyword_score": 3.8,
        }
    ]
    connect.assert_called_once_with(retrieval.get_database_url())
    cursor.execute.assert_called_once()

########################################
############### DENSE ##################
########################################

def test_vector_literal_formats_pgvector_value():
    assert retrieval.vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

def test_search_dense_papers_returns_empty_for_blank_query(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_texts", Mock())

    assert retrieval.search_dense_papers("   ") == []

    retrieval.embed_texts.assert_not_called()

def test_search_dense_papers_embeds_query_and_maps_database_rows(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_texts", Mock(return_value=[[0.1, 0.2]]))

    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("2604.25917", "Recursive Multi-Agent Systems", None, 0.87),
    ]

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(retrieval.psycopg, "connect", connect)

    results = retrieval.search_dense_papers(" recursive agents ", limit=5)

    assert results == [
        {
            "arxiv_id": "2604.25917",
            "title": "Recursive Multi-Agent Systems",
            "abstract": "",
            "dense_score": 0.87,
        }
    ]
    retrieval.embed_texts.assert_called_once_with(["recursive agents"])
    cursor.execute.assert_called_once()
    params = cursor.execute.call_args.args[1]
    assert params == ("[0.1,0.2]", "[0.1,0.2]", 5)

########################################
############### HYBRID #################
########################################

def test_normalize_score_divides_by_max_score():
    assert retrieval.normalize_score(2.0, 4.0) == 0.5

def test_normalize_score_returns_zero_when_max_score_is_zero():
    assert retrieval.normalize_score(2.0, 0.0) == 0.0

def test_search_hybrid_papers_merges_and_ranks_results(monkeypatch):
    keyword_results = [
        {
            "arxiv_id": "keyword-only",
            "title": "Keyword Match",
            "abstract": "",
            "keyword_score": 4.0,
        },
        {
            "arxiv_id": "both",
            "title": "Both Match",
            "abstract": "",
            "keyword_score": 2.0,
        },
    ]
    dense_results = [
        {
            "arxiv_id": "dense-only",
            "title": "Dense Match",
            "abstract": "",
            "dense_score": 0.9,
        },
        {
            "arxiv_id": "both",
            "title": "Both Match",
            "abstract": "",
            "dense_score": 0.45,
        },
    ]

    monkeypatch.setattr(retrieval, "search_keyword_papers", Mock(return_value=keyword_results))
    monkeypatch.setattr(retrieval, "search_dense_papers", Mock(return_value=dense_results))

    results = retrieval.search_hybrid_papers(
        "agents",
        limit=3,
        dense_weight=0.6,
        keyword_weight=0.4,
    )

    assert [result["arxiv_id"] for result in results] == [
        "dense-only",
        "both",
        "keyword-only",
    ]
    assert results[0]["final_score"] == 0.6
    assert results[1]["final_score"] == 0.5
    assert results[2]["final_score"] == 0.4

def test_search_hybrid_papers_requires_both_weights():
    with pytest.raises(ValueError, match="must be provided together"):
        retrieval.search_hybrid_papers("agents", dense_weight=0.7)

def test_search_hybrid_papers_uses_configured_weights_by_default(monkeypatch):
    keyword_results = [
        {
            "arxiv_id": "keyword-only",
            "title": "Keyword Match",
            "abstract": "",
            "keyword_score": 4.0,
        },
    ]
    dense_results = [
        {
            "arxiv_id": "dense-only",
            "title": "Dense Match",
            "abstract": "",
            "dense_score": 0.9,
        },
    ]

    monkeypatch.setattr(retrieval, "search_keyword_papers", Mock(return_value=keyword_results))
    monkeypatch.setattr(retrieval, "search_dense_papers", Mock(return_value=dense_results))
    monkeypatch.setattr(retrieval, "get_hybrid_weights", Mock(return_value=(0.25, 0.75)))

    results = retrieval.search_hybrid_papers("agents", limit=2)

    assert [result["arxiv_id"] for result in results] == [
        "keyword-only",
        "dense-only",
    ]
    assert results[0]["final_score"] == 0.75
    assert results[1]["final_score"] == 0.25
    retrieval.get_hybrid_weights.assert_called_once_with()
