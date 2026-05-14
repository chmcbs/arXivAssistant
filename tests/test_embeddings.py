"""
Tests the embeddings pipeline
"""

from unittest.mock import MagicMock, Mock
from core import embeddings
from core.embeddings import paper_text

def test_paper_text_combines_title_and_abstract():
    paper = {
        "title": "Title goes here",
        "abstract": "Abstract goes here",
    }

    assert paper_text(paper) == "Title goes here\n\nAbstract goes here"

def test_run_embeddings_returns_zero_when_no_papers(monkeypatch):
    monkeypatch.setattr(embeddings, "get_papers_missing_embeddings", Mock(return_value=[]))
    monkeypatch.setattr(embeddings, "embed_texts", Mock())
    monkeypatch.setattr(embeddings, "save_embeddings", Mock())

    saved_count = embeddings.run_embeddings(limit=10)

    assert saved_count == 0
    embeddings.get_papers_missing_embeddings.assert_called_once_with(10)
    embeddings.embed_texts.assert_not_called()
    embeddings.save_embeddings.assert_not_called()

def test_run_embeddings_generates_and_saves_embeddings(monkeypatch):
    papers = [
        {
            "arxiv_id": "2401.12345",
            "title": "First paper",
            "abstract": "First abstract",
        },
        {
            "arxiv_id": "2401.67890",
            "title": "Second paper",
            "abstract": "Second abstract",
        },
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(embeddings, "get_papers_missing_embeddings", Mock(return_value=papers))
    monkeypatch.setattr(embeddings, "embed_texts", Mock(return_value=vectors))
    monkeypatch.setattr(embeddings, "save_embeddings", Mock(return_value=2))

    saved_count = embeddings.run_embeddings(limit=10)

    assert saved_count == 2
    embeddings.get_papers_missing_embeddings.assert_called_once_with(10)
    embeddings.embed_texts.assert_called_once_with(
        [
            "First paper\n\nFirst abstract",
            "Second paper\n\nSecond abstract",
        ]
    )
    embeddings.save_embeddings.assert_called_once_with(papers, vectors)

def test_save_embeddings_maps_papers_and_vectors_to_database_rows(monkeypatch):
    papers = [
        {"arxiv_id": "2401.12345"},
        {"arxiv_id": "2401.67890"},
    ]
    vectors = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    cursor = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(embeddings.psycopg, "connect", connect)

    saved_count = embeddings.save_embeddings(papers, vectors)

    assert saved_count == 2
    cursor.executemany.assert_called_once()
    rows = cursor.executemany.call_args.args[1]
    assert rows == [
        {
            "arxiv_id": "2401.12345",
            "embedding": [0.1, 0.2],
            "model_name": embeddings.MODEL_NAME,
        },
        {
            "arxiv_id": "2401.67890",
            "embedding": [0.3, 0.4],
            "model_name": embeddings.MODEL_NAME,
        },
    ]