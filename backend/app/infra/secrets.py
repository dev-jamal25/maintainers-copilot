"""Runtime secret loading from Vault (CLAUDE.md: all runtime secrets come from Vault).

The JWT signing secret lives at ``secret/maintainers-copilot/jwt`` under key ``secret`` and the
Anthropic key at ``secret/maintainers-copilot/anthropic`` under ``api_key`` — exactly what
``scripts/vault-init.sh`` seeds. Reads go through the ``VaultClient`` adapter and are cached per
(addr, token) so request-time dependencies do not hammer Vault. The compose ``api`` boot step fails
loud if Vault is unreachable, so by the time requests are served the read succeeds.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import VaultSettings
from app.infra.errors import VaultSecretNotFoundError
from app.infra.vault import VaultClient

_JWT_VAULT_PATH = "maintainers-copilot/jwt"
_JWT_SECRET_KEY = "secret"  # noqa: S105 - Vault key name, not a secret value
_ANTHROPIC_VAULT_PATH = "maintainers-copilot/anthropic"
_ANTHROPIC_KEY = "api_key"


@lru_cache(maxsize=4)
def _read_secret_field(addr: str, token: str, path: str, key: str) -> str:
    client = VaultClient(VaultSettings(addr=addr, dev_root_token_id=token))
    secret = client.read_secret(path)
    value = secret.get(key, "")
    if not value:
        raise VaultSecretNotFoundError(f"Vault secret '{path}' is missing '{key}'")
    return value


def load_jwt_secret(settings: VaultSettings) -> str:
    """Return the JWT signing secret from Vault (cached). Raises VaultError on failure."""
    return _read_secret_field(
        settings.addr, settings.dev_root_token_id, _JWT_VAULT_PATH, _JWT_SECRET_KEY
    )


def load_anthropic_api_key(settings: VaultSettings) -> str:
    """Return the Anthropic API key from Vault (cached). Raises VaultError on failure."""
    return _read_secret_field(
        settings.addr, settings.dev_root_token_id, _ANTHROPIC_VAULT_PATH, _ANTHROPIC_KEY
    )
