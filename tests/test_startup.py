"""
Tests production startup configuration validation
"""

import pytest

from core.startup import StartupConfigError, validate_runtime_config


def _set_valid_production_env(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("DISABLE_CSRF", raising=False)
    monkeypatch.delenv("DISABLE_RATE_LIMIT", raising=False)
    monkeypatch.delenv("ALLOW_DEV_MAGIC_LINK_RESPONSE", raising=False)
    monkeypatch.setenv("INTERNAL_CRON_TOKEN", "x" * 32)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("EMAIL_FROM", "noreply@example.com")


def test_validate_runtime_config_allows_development_defaults(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DISABLE_CSRF", raising=False)

    validate_runtime_config()


def test_validate_runtime_config_rejects_disabled_csrf_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DISABLE_CSRF", "1")
    monkeypatch.setenv("INTERNAL_CRON_TOKEN", "x" * 32)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("EMAIL_FROM", "noreply@example.com")

    with pytest.raises(StartupConfigError, match="DISABLE_CSRF"):
        validate_runtime_config()


def test_validate_runtime_config_requires_strong_cron_token_in_production(monkeypatch):
    _set_valid_production_env(monkeypatch)
    monkeypatch.setenv("INTERNAL_CRON_TOKEN", "short")

    with pytest.raises(StartupConfigError, match="INTERNAL_CRON_TOKEN"):
        validate_runtime_config()


def test_validate_runtime_config_requires_email_delivery_in_production(monkeypatch):
    _set_valid_production_env(monkeypatch)
    monkeypatch.delenv("SMTP_HOST", raising=False)

    with pytest.raises(StartupConfigError, match="SMTP_HOST and EMAIL_FROM"):
        validate_runtime_config()


def test_validate_runtime_config_allows_valid_production_settings(monkeypatch):
    _set_valid_production_env(monkeypatch)

    validate_runtime_config()
