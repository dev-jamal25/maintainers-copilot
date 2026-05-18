from unittest.mock import MagicMock, patch

import pytest

from app.core.settings import VaultSettings
from app.core.startup_checks import check_vault_ready, main
from app.infra.errors import VaultConfigError, VaultUnavailableError


def _settings(addr: str = "http://vault:8200", auth_value: str | None = None) -> VaultSettings:
    resolved_auth_value = "unit-test-value" if auth_value is None else auth_value
    return VaultSettings(addr=addr, dev_root_token_id=resolved_auth_value)


def test_check_vault_ready_propagates_config_error() -> None:
    with pytest.raises(VaultConfigError):
        check_vault_ready(_settings(addr=""))


def test_check_vault_ready_succeeds_with_patched_client() -> None:
    with patch("app.core.startup_checks.VaultClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        check_vault_ready(_settings())
        instance.check_ready.assert_called_once()


def test_main_returns_zero_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.core.startup_checks.check_vault_ready") as mock_check:
        mock_check.return_value = None
        assert main() == 0
    captured = capsys.readouterr()
    assert "passed" in captured.out


def test_main_returns_one_on_vault_error(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.core.startup_checks.check_vault_ready") as mock_check:
        mock_check.side_effect = VaultUnavailableError("Vault unreachable at http://vault:8200")
        assert main() == 1
    captured = capsys.readouterr()
    assert "failed" in captured.err
    assert "unreachable" in captured.err
