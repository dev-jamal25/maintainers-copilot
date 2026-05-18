from unittest.mock import MagicMock

import hvac.exceptions
import pytest

from app.core.settings import VaultSettings
from app.infra.errors import (
    VaultAuthenticationError,
    VaultConfigError,
    VaultSecretNotFoundError,
    VaultUnavailableError,
)
from app.infra.vault import VaultClient


def _settings(addr: str = "http://vault:8200", auth_value: str | None = None) -> VaultSettings:
    resolved_auth_value = "unit-test-value" if auth_value is None else auth_value
    return VaultSettings(addr=addr, dev_root_token_id=resolved_auth_value)


def _ready_client() -> MagicMock:
    """Mock hvac.Client where seal_status is healthy and auth is valid."""
    m = MagicMock()
    m.url = "http://vault:8200"
    m.sys.read_seal_status.return_value = {"sealed": False, "initialized": True}
    m.is_authenticated.return_value = True
    return m


def test_check_ready_passes_when_unsealed_and_authenticated() -> None:
    client = VaultClient(_settings(), hvac_client=_ready_client())
    client.check_ready()


def test_missing_vault_addr_raises_config_error() -> None:
    with pytest.raises(VaultConfigError, match="VAULT_ADDR"):
        VaultClient(_settings(addr=""), hvac_client=MagicMock())


def test_missing_vault_token_raises_config_error() -> None:
    with pytest.raises(VaultConfigError, match="VAULT_DEV_ROOT_TOKEN_ID"):
        VaultClient(_settings(auth_value=""), hvac_client=MagicMock())


def test_unreachable_vault_raises_unavailable_error() -> None:
    m = MagicMock()
    m.url = "http://vault:8200"
    m.sys.read_seal_status.side_effect = ConnectionRefusedError("connection refused")
    client = VaultClient(_settings(), hvac_client=m)
    with pytest.raises(VaultUnavailableError, match="unreachable"):
        client.check_ready()


def test_sealed_vault_raises_unavailable_error() -> None:
    m = _ready_client()
    m.sys.read_seal_status.return_value = {"sealed": True, "initialized": True}
    client = VaultClient(_settings(), hvac_client=m)
    with pytest.raises(VaultUnavailableError, match="sealed"):
        client.check_ready()


def test_unauthenticated_client_raises_auth_error() -> None:
    m = _ready_client()
    m.is_authenticated.return_value = False
    client = VaultClient(_settings(), hvac_client=m)
    with pytest.raises(VaultAuthenticationError, match="invalid"):
        client.check_ready()


def test_read_secret_unwraps_kv_v2_envelope() -> None:
    m = _ready_client()
    m.secrets.kv.v2.read_secret_version.return_value = {
        "data": {"data": {"password": "example", "user": "svc"}},
    }
    client = VaultClient(_settings(), hvac_client=m)
    assert client.read_secret("maintainers-copilot/database") == {
        "password": "example",
        "user": "svc",
    }


def test_missing_secret_path_raises_not_found() -> None:
    m = _ready_client()
    m.secrets.kv.v2.read_secret_version.side_effect = hvac.exceptions.InvalidPath("nope")
    client = VaultClient(_settings(), hvac_client=m)
    with pytest.raises(VaultSecretNotFoundError, match="not found"):
        client.read_secret("maintainers-copilot/missing")
