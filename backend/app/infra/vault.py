import hvac
import hvac.exceptions

from app.core.settings import VaultSettings
from app.infra.errors import (
    VaultAuthenticationError,
    VaultConfigError,
    VaultSecretNotFoundError,
    VaultUnavailableError,
)


class VaultClient:
    """Thin adapter around hvac.Client for KV v2 secret retrieval.

    Hides hvac specifics so the rest of the app sees only the typed
    contract: check_ready() and read_secret(path).
    """

    _MOUNT_POINT = "secret"

    def __init__(
        self,
        settings: VaultSettings,
        *,
        hvac_client: hvac.Client | None = None,
    ) -> None:
        if not settings.addr:
            raise VaultConfigError("VAULT_ADDR is empty")
        if not settings.dev_root_token_id:
            raise VaultConfigError("VAULT_DEV_ROOT_TOKEN_ID is empty")

        self._client = hvac_client or hvac.Client(
            url=settings.addr,
            token=settings.dev_root_token_id,
        )

    def check_ready(self) -> None:
        """Raise if Vault is unreachable, sealed, uninitialized, or unauthenticated."""
        try:
            seal_status = self._client.sys.read_seal_status()
        except (OSError, hvac.exceptions.VaultError) as e:
            raise VaultUnavailableError(f"Vault unreachable at {self._client.url}") from e

        if seal_status.get("sealed"):
            raise VaultUnavailableError("Vault is sealed")
        if not seal_status.get("initialized"):
            raise VaultUnavailableError("Vault is not initialized")

        try:
            authenticated = self._client.is_authenticated()
        except hvac.exceptions.Forbidden as e:
            raise VaultAuthenticationError("Vault token is forbidden") from e
        except (OSError, hvac.exceptions.VaultError) as e:
            raise VaultUnavailableError(f"Vault unreachable at {self._client.url}") from e

        if not authenticated:
            raise VaultAuthenticationError("Vault token is invalid")

    def read_secret(self, path: str) -> dict[str, str]:
        """Read a KV v2 secret under the 'secret' mount and return its data dict.

        Hides the response['data']['data'] unwrapping that KV v2 requires.
        """
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self._MOUNT_POINT,
            )
        except hvac.exceptions.InvalidPath as e:
            raise VaultSecretNotFoundError(f"Vault path '{path}' not found") from e
        except hvac.exceptions.Forbidden as e:
            raise VaultAuthenticationError(f"Vault forbidden reading '{path}'") from e
        except (OSError, hvac.exceptions.VaultError) as e:
            raise VaultUnavailableError(f"Vault unreachable while reading '{path}'") from e

        try:
            data = response["data"]["data"]
        except (KeyError, TypeError) as e:
            raise VaultSecretNotFoundError(f"Vault path '{path}' returned no data envelope") from e

        if not isinstance(data, dict):
            raise VaultSecretNotFoundError(f"Vault path '{path}' returned non-dict data")

        return {str(k): str(v) for k, v in data.items()}
