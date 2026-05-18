from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings, TracingSettings, VaultSettings
from app.infra.errors import (
    TracingConfigError,
    TracingUnavailableError,
    VaultSecretNotFoundError,
)
from app.infra.tracing import (
    LangfuseTracingClient,
    NullTracingClient,
    build_tracing_client,
)

_HOST = "https://langfuse.example"
_PUBLIC_VALUE = "public-value"
_PRIVATE_VALUE = "LEAK-CANARY"


def _settings(
    tracing: TracingSettings | None = None,
) -> Settings:
    auth_value = "unit-test-value"
    return Settings(
        vault=VaultSettings(
            addr="http://vault:8200",
            dev_root_token_id=auth_value,
        ),
        tracing=tracing if tracing is not None else TracingSettings(enabled=True, host=_HOST),
    )


def _vault_client(secret: dict[str, str] | None = None) -> MagicMock:
    client = MagicMock()
    client.read_secret.return_value = secret if secret is not None else _valid_secret()
    return client


def _valid_secret() -> dict[str, str]:
    return {"public_key": _PUBLIC_VALUE, "secret_key": _PRIVATE_VALUE}


def test_disabled_tracing_returns_null_client() -> None:
    vault_client = _vault_client()
    client = build_tracing_client(
        _settings(TracingSettings(enabled=False, host="")),
        vault_client,
    )

    assert isinstance(client, NullTracingClient)
    vault_client.read_secret.assert_not_called()
    client.check_ready()


def test_enabled_tracing_builds_langfuse_client_from_vault_secret() -> None:
    vault_client = _vault_client()

    with patch("app.infra.tracing.langfuse.Langfuse") as MockLangfuse:
        langfuse_client = MagicMock()
        MockLangfuse.return_value = langfuse_client

        client = build_tracing_client(_settings(), vault_client)

    assert isinstance(client, LangfuseTracingClient)
    vault_client.read_secret.assert_called_once_with("maintainers-copilot/langfuse")
    MockLangfuse.assert_called_once_with(
        public_key=_PUBLIC_VALUE,
        secret_key=_PRIVATE_VALUE,
        host=_HOST,
    )


def test_missing_langfuse_host_raises_config_error() -> None:
    vault_client = _vault_client()

    with pytest.raises(TracingConfigError, match="LANGFUSE_HOST"):
        build_tracing_client(_settings(TracingSettings(enabled=True, host="")), vault_client)

    vault_client.read_secret.assert_not_called()


def test_missing_public_key_in_vault_secret_raises_config_error() -> None:
    vault_client = _vault_client({"secret_key": _PRIVATE_VALUE})

    with pytest.raises(TracingConfigError, match="public_key"):
        build_tracing_client(_settings(), vault_client)


def test_missing_secret_key_in_vault_secret_raises_config_error() -> None:
    vault_client = _vault_client({"public_key": _PUBLIC_VALUE})

    with pytest.raises(TracingConfigError, match="secret_key"):
        build_tracing_client(_settings(), vault_client)


def test_vault_missing_langfuse_path_propagates_not_found() -> None:
    vault_client = _vault_client()
    vault_client.read_secret.side_effect = VaultSecretNotFoundError("missing")

    with pytest.raises(VaultSecretNotFoundError, match="missing"):
        build_tracing_client(_settings(), vault_client)


def test_langfuse_init_failure_raises_unavailable_error() -> None:
    vault_client = _vault_client()

    with patch("app.infra.tracing.langfuse.Langfuse") as MockLangfuse:
        original = RuntimeError("init failed")
        MockLangfuse.side_effect = original

        with pytest.raises(TracingUnavailableError, match="client init failed") as exc_info:
            build_tracing_client(_settings(), vault_client)

    assert exc_info.value.__cause__ is original


def test_langfuse_check_ready_failure_raises_unavailable_error() -> None:
    langfuse_client = MagicMock()
    original = RuntimeError("flush failed")
    langfuse_client.flush.side_effect = original
    client = LangfuseTracingClient(langfuse_client)

    with pytest.raises(TracingUnavailableError, match="readiness check failed") as exc_info:
        client.check_ready()

    assert exc_info.value.__cause__ is original


def test_errors_never_contain_secret_values() -> None:
    error_messages: list[str] = []

    def capture_error(fn: Callable[[], object]) -> None:
        try:
            fn()
        except Exception as e:
            error_messages.append(str(e))
            if e.__cause__ is not None:
                error_messages.append(str(e.__cause__))

    capture_error(
        lambda: build_tracing_client(
            _settings(TracingSettings(enabled=True, host="")),
            _vault_client(),
        )
    )
    capture_error(
        lambda: build_tracing_client(
            _settings(),
            _vault_client({"secret_key": _PRIVATE_VALUE}),
        )
    )
    capture_error(
        lambda: build_tracing_client(
            _settings(),
            _vault_client({"public_key": _PUBLIC_VALUE}),
        )
    )

    vault_client = _vault_client()
    with patch("app.infra.tracing.langfuse.Langfuse", side_effect=RuntimeError("init failed")):
        capture_error(lambda: build_tracing_client(_settings(), vault_client))

    langfuse_client = MagicMock()
    langfuse_client.flush.side_effect = RuntimeError("flush failed")
    capture_error(lambda: LangfuseTracingClient(langfuse_client).check_ready())

    assert error_messages
    assert all(_PRIVATE_VALUE not in message for message in error_messages)
