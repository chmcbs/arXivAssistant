"""
Shared pytest fixtures for the test suite
"""

import pytest


@pytest.fixture(autouse=True)
def _no_auth_db_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.routes.get_auth_session_payload",
        lambda *_args, **_kwargs: {
            "authenticated": False,
            "user_id": None,
            "email": None,
        },
    )
