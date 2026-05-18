import pytest

from app.core.config import DatabaseSettings, Settings


def test_database_settings_has_local_dev_default() -> None:
    settings = DatabaseSettings()
    assert settings.database_url == (
        "postgresql+asyncpg://maintainers_copilot@db:5432/maintainers_copilot"
    )


def test_database_settings_reads_database_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://other_user@other_host:5433/other_db",
    )
    settings = DatabaseSettings()
    assert settings.database_url == ("postgresql+asyncpg://other_user@other_host:5433/other_db")


def test_settings_aggregate_includes_database() -> None:
    settings = Settings()
    assert settings.database.database_url.startswith("postgresql+asyncpg://")


def test_database_url_uses_async_driver() -> None:
    assert DatabaseSettings().database_url.startswith("postgresql+asyncpg://")
