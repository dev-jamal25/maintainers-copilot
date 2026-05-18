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
