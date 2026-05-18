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


@dataclass
class Settings:
    """Aggregated app settings; built from env-derived sub-settings.

    Lifespan callers pass this whole object so future settings groups
    (db, redis, minio) can attach here without changing call sites.
    """

    vault: VaultSettings = field(default_factory=VaultSettings)
    tracing: TracingSettings = field(default_factory=TracingSettings)
