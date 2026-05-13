"""
Tests user profile helpers
"""

from unittest.mock import MagicMock, Mock
import pytest
import profiles

def _mock_connection_with_cursor(cursor):
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock()
    connect.return_value.__enter__.return_value = connection
    return connect

def test_pick_next_available_slot_returns_first_gap():
    assert profiles._pick_next_available_slot({1, 3}) == 2

def test_pick_next_available_slot_raises_when_cap_is_reached():
    with pytest.raises(ValueError, match="profile cap"):
        profiles._pick_next_available_slot({1, 2, 3})

def test_validate_interest_sentence_rejects_blank():
    with pytest.raises(ValueError, match="interest_sentence must not be empty"):
        profiles._validate_interest_sentence("   ")

def test_validate_category_rejects_non_configured_values(monkeypatch):
    monkeypatch.setattr(profiles, "get_arxiv_categories", Mock(return_value=["cs.AI"]))

    with pytest.raises(ValueError, match="configured ARXIV_CATEGORIES"):
        profiles._validate_category("cs.CL")

def test_create_profile_inserts_with_next_slot(monkeypatch):
    monkeypatch.setattr(profiles.uuid, "uuid4", Mock(return_value="profile-123"))
    monkeypatch.setattr(profiles, "get_arxiv_categories", Mock(return_value=["cs.AI"]))

    cursor = MagicMock()
    cursor.fetchall.return_value = [(1,)]  # slot 1 taken -> choose slot 2
    monkeypatch.setattr(profiles.psycopg, "connect", _mock_connection_with_cursor(cursor))

    profile_id = profiles.create_profile(
        user_id="user-1",
        category="cs.AI",
        interest_sentence="Language model planning",
    )

    assert profile_id == "profile-123"
    assert cursor.execute.call_count == 2

    insert_params = cursor.execute.call_args_list[1].args[1]
    assert insert_params == (
        "profile-123",
        "user-1",
        2,
        "cs.AI",
        "Language model planning",
    )

def test_create_profile_raises_when_user_has_three_profiles(monkeypatch):
    monkeypatch.setattr(profiles, "get_arxiv_categories", Mock(return_value=["cs.AI"]))

    cursor = MagicMock()
    cursor.fetchall.return_value = [(1,), (2,), (3,)]
    monkeypatch.setattr(profiles.psycopg, "connect", _mock_connection_with_cursor(cursor))

    with pytest.raises(ValueError, match="3-profile cap"):
        profiles.create_profile(user_id="user-1", category="cs.AI", interest_sentence="test")

def test_list_profiles_maps_rows_to_dicts(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("p-1", "user-1", 1, "cs.AI", "Interest A", "2026-01-01T00:00:00Z"),
        ("p-2", "user-1", 2, "cs.CL", "Interest B", "2026-01-02T00:00:00Z"),
    ]
    monkeypatch.setattr(profiles.psycopg, "connect", _mock_connection_with_cursor(cursor))

    results = profiles.list_profiles(user_id="user-1")

    assert [item["profile_id"] for item in results] == ["p-1", "p-2"]
    assert results[0]["profile_slot"] == 1
    assert results[1]["category"] == "cs.CL"

def test_get_profile_returns_none_when_not_found(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    monkeypatch.setattr(profiles.psycopg, "connect", _mock_connection_with_cursor(cursor))

    assert profiles.get_profile("missing") is None

def test_get_or_create_default_profile_returns_existing_profile(monkeypatch):
    monkeypatch.setattr(
        profiles,
        "list_profiles",
        Mock(return_value=[{"profile_id": "p-1", "profile_slot": 1}]),
    )
    monkeypatch.setattr(profiles, "create_profile", Mock())

    result = profiles.get_or_create_default_profile(user_id="user-1")

    assert result["profile_id"] == "p-1"
    profiles.create_profile.assert_not_called()

def test_get_or_create_default_profile_creates_when_missing(monkeypatch):
    monkeypatch.setattr(profiles, "list_profiles", Mock(return_value=[]))
    monkeypatch.setattr(profiles, "create_profile", Mock(return_value="p-new"))
    monkeypatch.setattr(
        profiles,
        "get_profile",
        Mock(return_value={"profile_id": "p-new", "profile_slot": 1}),
    )

    result = profiles.get_or_create_default_profile(
        user_id="user-1",
        category="cs.AI",
        interest_sentence="Interest",
    )

    assert result["profile_id"] == "p-new"
    profiles.create_profile.assert_called_once_with(
        user_id="user-1",
        category="cs.AI",
        interest_sentence="Interest",
    )
