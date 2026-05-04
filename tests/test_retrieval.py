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