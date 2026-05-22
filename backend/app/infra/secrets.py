"""Runtime secret loading from Vault (CLAUDE.md: all runtime secrets come from Vault).

The JWT signing secret lives at ``secret/maintainers-copilot/auth`` under key ``jwt_secret``. It is
read through the existing ``VaultClient`` adapter and cached per (addr, token) so request-time
dependencies do not hammer Vault. The compose ``api`` boot step already fails loud if Vault is
unreachable, so by the time requests are served the read succeeds.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import VaultSettings
from app.infra.errors import VaultSecretNotFoundError
from app.infra.vault import VaultClient

_AUTH_VAULT_PATH = "maintainers-copilot/auth"
_JWT_SECRET_KEY = "jwt_secret"  # noqa: S105 - Vault key name, not a secret value


@lru_cache(maxsize=4)
def _read_jwt_secret(addr: str, token: str) -> str:
    client = VaultClient(VaultSettings(addr=addr, dev_root_token_id=token))
    secret = client.read_secret(_AUTH_VAULT_PATH)
    value = secret.get(_JWT_SECRET_KEY, "")
    if not value:
        raise VaultSecretNotFoundError(
            f"Vault secret '{_AUTH_VAULT_PATH}' is missing '{_JWT_SECRET_KEY}'"
        )
    return value


def load_jwt_secret(settings: VaultSettings) -> str:
    """Return the JWT signing secret from Vault (cached). Raises VaultError on failure."""
    return _read_jwt_secret(settings.addr, settings.dev_root_token_id)
