"""
Tests recommendation generation and persistence
"""
from unittest.mock import MagicMock, Mock
import pytest
import recommendations

def _mock_connection_with_cursor(cursor):
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    return connect, cursor

def test_generate_recommendations_requires_completed_run(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    connect, _ = _mock_connection_with_cursor(cursor)
    monkeypatch.setattr(recommendations.psycopg, "connect", connect)

    with pytest.raises(ValueError, match="must exist and be completed"):
        recommendations.generate_recommendations("run-123", user_id="default")

def test_generate_recommendations_replaces_rows_deterministically(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        ("run-123", "cs.AI", 150),
        (3,),
    ]
    cursor.fetchall.return_value = [
        (
            1,
            "2601.00001",
            "Paper A",
            None,
            0,
            "run",
            0.9,
        ),
        (
            2,
            "2601.00002",
            "Paper B",
            "abstract",
            1,
            "7d",
            0.8,
        ),
    ]

    connect, _ = _mock_connection_with_cursor(cursor)
    monkeypatch.setattr(recommendations.psycopg, "connect", connect)
    monkeypatch.setattr(recommendations, "get_daily_picks_k", Mock(return_value=3))
    monkeypatch.setattr(
        recommendations.uuid,
        "uuid4",
        Mock(side_effect=["rec-1", "rec-2"]),
    )

    results = recommendations.generate_recommendations("run-123", user_id="default")

    assert [result["rank"] for result in results] == [1, 2]
    assert [result["arxiv_id"] for result in results] == ["2601.00001", "2601.00002"]
    assert cursor.executemany.call_count == 1
    assert cursor.execute.call_count == 4

    delete_params = cursor.execute.call_args_list[3].args[1]
    assert delete_params == ("run-123", "default")

    inserted_rows = cursor.executemany.call_args.args[1]
    assert inserted_rows[0][0] == "rec-1"
    assert inserted_rows[0][3] == "2601.00001"
    assert inserted_rows[0][4] == 1
    assert inserted_rows[0][5] == 0.9
    assert inserted_rows[0][6] == "run"
    assert inserted_rows[0][7] == 0

def test_generate_recommendations_respects_k_override(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("run-123", "cs.AI", 150)]
    cursor.fetchall.return_value = []

    connect, _ = _mock_connection_with_cursor(cursor)
    monkeypatch.setattr(recommendations.psycopg, "connect", connect)
    monkeypatch.setattr(recommendations, "get_daily_picks_k", Mock(return_value=3))

    recommendations.generate_recommendations(
        "run-123",
        user_id="default",
        k_override=2,
    )

    rank_params = cursor.execute.call_args_list[1].args[1]
    assert rank_params == ("run-123", "default", "default", "default", 2)

def test_generate_recommendations_rejects_invalid_override(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("run-123", "cs.AI", 150)]

    connect, _ = _mock_connection_with_cursor(cursor)
    monkeypatch.setattr(recommendations.psycopg, "connect", connect)

    with pytest.raises(ValueError, match="k_override must be >= 1"):
        recommendations.generate_recommendations(
            "run-123",
            user_id="default",
            k_override=0,
        )