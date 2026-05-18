from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings, TracingSettings, VaultSettings
from app.core.lifespan import main, verify_startup_dependencies
from app.infra.errors import (
    TracingConfigError,
    TracingUnavailableError,
    VaultConfigError,
    VaultUnavailableError,
)


def _vault_settings(
    addr: str = "http://vault:8200",
    auth_value: str | None = None,
) -> VaultSettings:
    resolved_auth_value = "unit-test-value" if auth_value is None else auth_value
    return VaultSettings(addr=addr, dev_root_token_id=resolved_auth_value)


def _settings(
    *,
    vault: VaultSettings | None = None,
    tracing: TracingSettings | None = None,
) -> Settings:
    return Settings(
        vault=vault if vault is not None else _vault_settings(),
        tracing=tracing
        if tracing is not None
        else TracingSettings(enabled=True, host="https://langfuse.example"),
    )


def test_verify_startup_dependencies_propagates_vault_config_error() -> None:
    with pytest.raises(VaultConfigError):
        verify_startup_dependencies(_settings(vault=_vault_settings(addr="")))


def test_verify_startup_dependencies_runs_vault_then_tracing_in_order() -> None:
    events: list[str] = []
    tracing_client = MagicMock()

    def record_tracing_check() -> None:
        events.append("tracing-check")

    def build_tracing_client(*_args: object) -> MagicMock:
        events.append("build-tracing")
        return tracing_client

    tracing_client.check_ready.side_effect = record_tracing_check

    with (
        patch("app.core.lifespan.VaultClient") as MockVaultClient,
        patch("app.core.lifespan.build_tracing_client") as mock_build_tracing_client,
    ):
        vault_client = MagicMock()

        def record_vault_check() -> None:
            events.append("vault-check")

        vault_client.check_ready.side_effect = record_vault_check
        MockVaultClient.return_value = vault_client
        mock_build_tracing_client.side_effect = build_tracing_client

        settings = _settings()
        verify_startup_dependencies(settings)

    MockVaultClient.assert_called_once_with(settings.vault)
    vault_client.check_ready.assert_called_once()
    mock_build_tracing_client.assert_called_once_with(settings, vault_client)
    tracing_client.check_ready.assert_called_once()
    assert events == ["vault-check", "build-tracing", "tracing-check"]


def test_verify_startup_dependencies_does_not_call_tracing_when_vault_fails() -> None:
    with (
        patch("app.core.lifespan.VaultClient") as MockVaultClient,
        patch("app.core.lifespan.build_tracing_client") as mock_build_tracing_client,
    ):
        vault_client = MagicMock()
        vault_client.check_ready.side_effect = VaultUnavailableError("Vault is sealed")
        MockVaultClient.return_value = vault_client

        with pytest.raises(VaultUnavailableError, match="sealed"):
            verify_startup_dependencies(_settings())

    mock_build_tracing_client.assert_not_called()


def test_verify_startup_dependencies_propagates_tracing_config_error() -> None:
    with (
        patch("app.core.lifespan.VaultClient") as MockVaultClient,
        patch("app.core.lifespan.build_tracing_client") as mock_build_tracing_client,
    ):
        MockVaultClient.return_value = MagicMock()
        mock_build_tracing_client.side_effect = TracingConfigError("LANGFUSE_HOST is empty")

        with pytest.raises(TracingConfigError, match="LANGFUSE_HOST"):
            verify_startup_dependencies(_settings())


def test_main_returns_zero_on_success_and_prints_both_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("app.core.lifespan.verify_startup_dependencies") as mock_check:
        mock_check.return_value = None
        assert main() == 0
    captured = capsys.readouterr()
    assert "Vault startup check passed" in captured.out
    assert "Tracing startup check passed" in captured.out


def test_main_returns_one_on_vault_error_and_prints_only_vault_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("app.core.lifespan.verify_startup_dependencies") as mock_check:
        mock_check.side_effect = VaultUnavailableError("Vault unreachable at http://vault:8200")
        assert main() == 1
    captured = capsys.readouterr()
    assert "Vault startup check failed" in captured.err
    assert "unreachable" in captured.err
    assert "Vault startup check passed" not in captured.out
    assert "Tracing startup check passed" not in captured.out


def test_main_returns_one_on_tracing_error_and_prints_only_tracing_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("app.core.lifespan.verify_startup_dependencies") as mock_check:
        mock_check.side_effect = TracingUnavailableError("Langfuse readiness check failed")
        assert main() == 1
    captured = capsys.readouterr()
    assert "Tracing startup check failed" in captured.err
    assert "Langfuse readiness check failed" in captured.err
    assert "Vault startup check passed" not in captured.out
    assert "Tracing startup check passed" not in captured.out


def test_main_with_tracing_disabled_still_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(tracing=TracingSettings(enabled=False, host=""))

    with (
        patch("app.core.lifespan.Settings", return_value=settings),
        patch("app.core.lifespan.VaultClient") as MockVaultClient,
    ):
        vault_client = MagicMock()
        MockVaultClient.return_value = vault_client

        assert main() == 0

    vault_client.check_ready.assert_called_once()
    vault_client.read_secret.assert_not_called()
    captured = capsys.readouterr()
    assert "Vault startup check passed" in captured.out
    assert "Tracing startup check passed" in captured.out
