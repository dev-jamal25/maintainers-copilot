from unittest.mock import MagicMock, patch

import pytest

from app.core.config import VaultSettings
from app.core.lifespan import main, verify_startup_dependencies
from app.infra.errors import VaultConfigError, VaultUnavailableError


def _settings(addr: str = "http://vault:8200", auth_value: str | None = None) -> VaultSettings:
    resolved_auth_value = "unit-test-value" if auth_value is None else auth_value
    return VaultSettings(addr=addr, dev_root_token_id=resolved_auth_value)


def test_verify_startup_dependencies_propagates_config_error() -> None:
    with pytest.raises(VaultConfigError):
        verify_startup_dependencies(_settings(addr=""))


def test_verify_startup_dependencies_succeeds_with_patched_client() -> None:
    with patch("app.core.lifespan.VaultClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        verify_startup_dependencies(_settings())
        instance.check_ready.assert_called_once()


def test_main_returns_zero_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.core.lifespan.verify_startup_dependencies") as mock_check:
        mock_check.return_value = None
        assert main() == 0
    captured = capsys.readouterr()
    assert "passed" in captured.out


def test_main_returns_one_on_vault_error(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.core.lifespan.verify_startup_dependencies") as mock_check:
        mock_check.side_effect = VaultUnavailableError("Vault unreachable at http://vault:8200")
        assert main() == 1
    captured = capsys.readouterr()
    assert "failed" in captured.err
    assert "unreachable" in captured.err
