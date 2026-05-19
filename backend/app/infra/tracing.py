from typing import Protocol

import langfuse

from app.core.config import Settings
from app.infra.errors import TracingConfigError, TracingUnavailableError
from app.infra.vault import VaultClient

_VAULT_LANGFUSE_PATH = "maintainers-copilot/langfuse"


class TracingClient(Protocol):
    """Stable surface the app uses; hides Langfuse SDK specifics."""

    def check_ready(self) -> None: ...

    def start_span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> object | None: ...

    def flush(self) -> None: ...


class NullTracingClient:
    """No-op client used when TRACING_ENABLED=false (local-dev escape hatch only)."""

    def check_ready(self) -> None:
        return None

    def start_span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> object | None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracingClient:
    """Real adapter. Wraps the Langfuse SDK so app code never imports langfuse."""

    def __init__(self, langfuse_client: langfuse.Langfuse) -> None:
        self._langfuse = langfuse_client

    def check_ready(self) -> None:
        try:
            self._langfuse.flush()
        except Exception as e:
            raise TracingUnavailableError("Langfuse readiness check failed") from e

    def start_span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> object | None:
        return None

    def flush(self) -> None:
        self._langfuse.flush()


def build_tracing_client(
    settings: Settings,
    vault_client: VaultClient,
) -> TracingClient:
    """Build a TracingClient based on settings."""
    if not settings.tracing.enabled:
        return NullTracingClient()

    if not settings.tracing.host:
        raise TracingConfigError("LANGFUSE_HOST is empty")

    secret = vault_client.read_secret(_VAULT_LANGFUSE_PATH)
    public_key = secret.get("public_key", "")
    secret_key = secret.get("secret_key", "")
    if not public_key:
        raise TracingConfigError(
            f"Langfuse secret at '{_VAULT_LANGFUSE_PATH}' is missing 'public_key'"
        )
    if not secret_key:
        raise TracingConfigError(
            f"Langfuse secret at '{_VAULT_LANGFUSE_PATH}' is missing 'secret_key'"
        )

    try:
        langfuse_client = langfuse.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.tracing.host,
        )
    except Exception as e:
        raise TracingUnavailableError(
            f"Langfuse client init failed for host '{settings.tracing.host}'"
        ) from e

    return LangfuseTracingClient(langfuse_client)
