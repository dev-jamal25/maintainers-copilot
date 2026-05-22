"""Settings groups added for Day 4: sane non-secret defaults + env-alias overrides."""

from __future__ import annotations

import pytest

from app.core.config import (
    AuthSettings,
    LLMSettings,
    ModelServerSettings,
    RedisSettings,
    Settings,
)


def test_redis_defaults() -> None:
    settings = RedisSettings()
    assert settings.url.startswith("redis://")
    assert settings.short_term_ttl_seconds == 86_400  # 24h MVP default (D4)


def test_redis_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_SHORT_TERM_TTL_SECONDS", "3600")
    assert RedisSettings().short_term_ttl_seconds == 3600


def test_model_server_defaults() -> None:
    settings = ModelServerSettings()
    assert settings.base_url.endswith(":8001")
    assert settings.timeout_seconds > 0
    assert settings.max_retries >= 1


def test_llm_uses_frozen_chat_model() -> None:
    assert LLMSettings().model == "claude-haiku-4-5-20251001"


def test_auth_defaults_have_no_secret_field() -> None:
    settings = AuthSettings()
    assert settings.jwt_lifetime_seconds == 3_600
    # The JWT signing secret must come from Vault, never settings.
    assert not any("secret" in name.lower() for name in AuthSettings.model_fields)


def test_settings_aggregate_includes_new_groups() -> None:
    settings = Settings()
    assert settings.redis.short_term_ttl_seconds > 0
    assert settings.model_server.base_url
    assert settings.llm.model
    assert settings.minio.bucket
    assert settings.auth.jwt_lifetime_seconds > 0
