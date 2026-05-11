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
############### SPARSE #################
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

########################################
########## PERSONALIZATION #############
########################################

def test_cosine_similarity_scores_vector_alignment():
    assert retrieval.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert retrieval.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert retrieval.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert retrieval.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_normalize_cosine_score_maps_cosine_range_to_score_range():
    assert retrieval.normalize_cosine_score(-1.0) == 0.0
    assert retrieval.normalize_cosine_score(0.0) == 0.5
    assert retrieval.normalize_cosine_score(1.0) == 1.0

def test_get_user_preference_vector_returns_none_when_missing(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(retrieval.psycopg, "connect", connect)

    assert retrieval.get_user_preference_vector("default") is None
    cursor.execute.assert_called_once()

    params = cursor.execute.call_args.args[1]
    assert params == ("default",)

def test_get_user_preference_vector_maps_database_row(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = ([0.1, 0.2],)

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(retrieval.psycopg, "connect", connect)

    assert retrieval.get_user_preference_vector("default") == [0.1, 0.2]

    params = cursor.execute.call_args.args[1]
    assert params == ("default",)

def test_get_paper_embedding_map_maps_database_rows(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("paper-a", [1.0, 0.0]),
        ("paper-b", [0.0, 1.0]),
    ]

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(retrieval.psycopg, "connect", connect)

    embedding_map = retrieval.get_paper_embedding_map(["paper-a", "paper-b"])

    assert embedding_map == {
        "paper-a": [1.0, 0.0],
        "paper-b": [0.0, 1.0],
    }

    params = cursor.execute.call_args.args[1]
    assert params == (["paper-a", "paper-b"],)

def test_search_personalized_papers_reranks_hybrid_candidates(monkeypatch):
    candidates = [
        {
            "arxiv_id": "hybrid-best",
            "title": "Hybrid Best",
            "abstract": "",
            "final_score": 1.0,
            "dense_score_norm": 1.0,
            "keyword_score_norm": 1.0,
        },
        {
            "arxiv_id": "preference-best",
            "title": "Preference Best",
            "abstract": "",
            "final_score": 0.5,
            "dense_score_norm": 0.5,
            "keyword_score_norm": 0.5,
        },
    ]

    monkeypatch.setattr(retrieval, "search_hybrid_papers", Mock(return_value=candidates))
    monkeypatch.setattr(retrieval, "get_user_preference_vector", Mock(return_value=[1.0, 0.0]))
    monkeypatch.setattr(
        retrieval,
        "get_paper_embedding_map",
        Mock(
            return_value={
                "hybrid-best": [-1.0, 0.0],
                "preference-best": [1.0, 0.0],
            }
        ),
    )

    results = retrieval.search_personalized_papers(
        "agents",
        user_id="default",
        limit=2,
        candidate_multiplier=3,
        hybrid_weight=0.5,
        preference_weight=0.5,
    )

    assert [result["arxiv_id"] for result in results] == [
        "preference-best",
        "hybrid-best",
    ]
    assert results[0]["preference_similarity"] == 1.0
    assert results[0]["preference_score"] == 1.0
    assert results[0]["personalized_score"] == 0.75
    retrieval.search_hybrid_papers.assert_called_once_with("agents", limit=6)
    retrieval.get_user_preference_vector.assert_called_once_with("default")

def test_search_personalized_papers_falls_back_to_hybrid_without_preference(monkeypatch):
    candidates = [
        {"arxiv_id": "a", "final_score": 1.0},
        {"arxiv_id": "b", "final_score": 0.9},
        {"arxiv_id": "c", "final_score": 0.8},
    ]

    monkeypatch.setattr(retrieval, "search_hybrid_papers", Mock(return_value=candidates))
    monkeypatch.setattr(retrieval, "get_user_preference_vector", Mock(return_value=None))
    monkeypatch.setattr(retrieval, "get_paper_embedding_map", Mock())

    results = retrieval.search_personalized_papers(
        "agents",
        user_id="default",
        limit=2,
    )

    assert results == candidates[:2]
    retrieval.get_paper_embedding_map.assert_not_called()

def test_search_personalized_papers_requires_non_negative_weights():
    with pytest.raises(ValueError, match="must be non-negative"):
        retrieval.search_personalized_papers("agents", hybrid_weight=-0.1)

def test_search_personalized_papers_requires_positive_total_weight():
    with pytest.raises(ValueError, match="At least one"):
        retrieval.search_personalized_papers(
            "agents",
            hybrid_weight=0.0,
            preference_weight=0.0,
        )