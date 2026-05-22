import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import langfuse

from app.core.config import Settings
from app.infra.errors import TracingConfigError, TracingUnavailableError
from app.infra.vault import VaultClient

logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def traced(
    client: TracingClient | None,
    name: str,
    *,
    trace_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AsyncIterator[None]:
    """Best-effort span around an operation (CLAUDE.md: span LLM/tool/RAG/rerank/memory calls).

    No-ops for ``None`` / ``NullTracingClient`` and never raises — observability must not break a
    request. This is the single hook the services use; when ``start_span`` gains real nesting the
    call sites do not change.
    """
    if client is not None:
        try:
            client.start_span(name, trace_id=trace_id, metadata=metadata)
        except Exception:  # noqa: BLE001 - tracing must never break the request path
            logger.debug("tracing span failed for %s", name, exc_info=True)
    yield


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
