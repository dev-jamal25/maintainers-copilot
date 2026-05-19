from dataclasses import dataclass, field

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VaultSettings(BaseSettings):
    """Vault bootstrap settings loaded from environment.

    Empty values are accepted at construction time and validated by
    VaultClient/startup checks, so settings can be inspected without
    raising. This keeps env loading separate from policy enforcement.
    """

    model_config = SettingsConfigDict(
        env_prefix="VAULT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    addr: str = Field(default="", description="Vault address, e.g. http://vault:8200")
    dev_root_token_id: str = Field(default="", description="Vault token (local dev only)")


class TracingSettings(BaseSettings):
    """Tracing config loaded from env. Non-secret only.

    Secrets (Langfuse public_key / secret_key) live in Vault at
    secret/maintainers-copilot/langfuse and are never read here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = Field(
        default=True,
        validation_alias="TRACING_ENABLED",
        description="Whether tracing is enabled at startup.",
    )
    host: str = Field(
        default="",
        validation_alias="LANGFUSE_HOST",
        description="Langfuse host URL, e.g. https://cloud.langfuse.com",
    )


class DatabaseSettings(BaseSettings):
    """Database connection config. Local-dev URL is passwordless.

    Real production-shaped credentials will resolve from Vault in a later
    chore; for now compose runs Postgres in trust mode and the local URL
    intentionally has no password embedded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(
        default="postgresql+asyncpg://maintainers_copilot@db:5432/maintainers_copilot",
        validation_alias="DATABASE_URL",
        description="Async SQLAlchemy URL; password must come from Vault later.",
    )


@dataclass
class Settings:
    """Aggregated app settings; built from env-derived sub-settings.

    Lifespan callers pass this whole object so future settings groups
    (redis, minio) can attach here without changing call sites.
    """

    vault: VaultSettings = field(default_factory=VaultSettings)
    tracing: TracingSettings = field(default_factory=TracingSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
