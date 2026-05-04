"""
Tests the retrieval pipeline
"""

from unittest.mock import MagicMock, Mock
import retrieval
from retrieval import keyword_query

def test_keyword_query_strips_whitespace():
    assert keyword_query("  recursive agents  ") == "recursive agents"

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

def test_vector_literal_formats_pgvector_value():
    assert retrieval.vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

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