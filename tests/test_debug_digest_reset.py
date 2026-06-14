from unittest.mock import Mock

from api.services.debug_digest_reset import reset_papers_and_runs
from core.preferences import RESET_ALL_PREFERENCE_EMBEDDINGS_SQL, reset_all_preference_embeddings


def test_reset_all_preference_embeddings_updates_all_profiles():
    cursor = Mock()
    cursor.rowcount = 3
    conn = Mock()
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)

    count = reset_all_preference_embeddings(conn)

    cursor.execute.assert_called_once_with(RESET_ALL_PREFERENCE_EMBEDDINGS_SQL)
    assert count == 3


def test_reset_papers_and_runs_resets_preference_embeddings(monkeypatch):
    cursor = Mock()
    cursor.fetchone.side_effect = [(2,), (5,)]
    conn = Mock()
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        "api.services.debug_digest_reset.reset_all_preference_embeddings",
        Mock(return_value=4),
    )

    result = reset_papers_and_runs(conn)

    assert result == {
        "deleted_runs": 2,
        "deleted_papers": 5,
        "reset_preference_embeddings": 4,
    }
    conn.commit.assert_called_once()
