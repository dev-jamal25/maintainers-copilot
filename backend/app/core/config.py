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


class RedisSettings(BaseSettings):
    """Short-term conversation-state store config (non-secret).

    TTL default is 24h (D4): long enough for a maintainer to resume a triage session the next day,
    short enough that abandoned anonymous sessions self-evict and Redis stays bounded. Long-term
    facts that must outlive this go through the explicit ``write_memory`` tool (Postgres).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    url: str = Field(
        default="redis://redis:6379/0",
        validation_alias="REDIS_URL",
        description="Redis connection URL for short-term conversation state.",
    )
    short_term_ttl_seconds: int = Field(
        default=86_400,
        validation_alias="REDIS_SHORT_TERM_TTL_SECONDS",
        description="TTL for chat:session:* keys (default 24h).",
    )


class ModelServerSettings(BaseSettings):
    """HTTP client config for the model-server boundary (classify/summarize/embed/rerank)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    base_url: str = Field(
        default="http://model-server:8001",
        validation_alias="MODEL_SERVER_URL",
        description="Base URL of the model-server FastAPI container.",
    )
    timeout_seconds: float = Field(default=30.0, validation_alias="MODEL_SERVER_TIMEOUT_SECONDS")
    max_retries: int = Field(default=2, validation_alias="MODEL_SERVER_MAX_RETRIES")


class LLMSettings(BaseSettings):
    """Chatbot LLM config (non-secret). The Anthropic API key is loaded from Vault at startup."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    model: str = Field(
        default="claude-haiku-4-5-20251001",
        validation_alias="CHAT_LLM_MODEL",
        description="Frozen chatbot model (DECISIONS: Claude Haiku 4.5).",
    )
    max_tokens: int = Field(default=1024, validation_alias="CHAT_LLM_MAX_TOKENS")
    timeout_seconds: float = Field(default=60.0, validation_alias="CHAT_LLM_TIMEOUT_SECONDS")


class MinioSettings(BaseSettings):
    """Object-store config (non-secret). Access/secret keys are loaded from Vault at startup."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    endpoint: str = Field(default="minio:9000", validation_alias="MINIO_ENDPOINT")
    bucket: str = Field(default="maintainers-copilot", validation_alias="MINIO_BUCKET")
    secure: bool = Field(default=False, validation_alias="MINIO_SECURE")


class AuthSettings(BaseSettings):
    """Auth config (non-secret). The JWT signing secret loads from Vault at startup, never here."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    jwt_lifetime_seconds: int = Field(default=3_600, validation_alias="AUTH_JWT_LIFETIME_SECONDS")
    invite_lifetime_seconds: int = Field(
        default=604_800,  # 7 days
        validation_alias="AUTH_INVITE_LIFETIME_SECONDS",
        description="How long an admin invite token stays valid.",
    )


@dataclass
class Settings:
    """Aggregated app settings; built from env-derived sub-settings.

    Lifespan callers pass this whole object so settings groups attach here without changing
    call sites. Secrets (Vault token excepted) are never stored on these settings — adapters
    resolve them from Vault at startup.
    """

    vault: VaultSettings = field(default_factory=VaultSettings)
    tracing: TracingSettings = field(default_factory=TracingSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    model_server: ModelServerSettings = field(default_factory=ModelServerSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    minio: MinioSettings = field(default_factory=MinioSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
