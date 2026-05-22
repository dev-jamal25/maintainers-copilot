"""FastAPI lifespan: build shared resources once and expose them via app.state.

Heavy/connection-holding resources (DB engine + session factory) are created here, not at import
time (CLAUDE.md: no expensive module-level globals). Fail-fast dependency checks (Vault, tracing)
remain the separate ``python -m app.core.lifespan`` boot step the compose ``api`` service runs
before launching uvicorn, so this lifespan stays import-safe and testable offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings
from app.db.session import create_engine, create_session_factory
from app.infra.errors import TracingError, VaultError
from app.infra.model_server_client import ModelServerClient
from app.infra.tracing import NullTracingClient, TracingClient, build_tracing_client
from app.infra.vault import VaultClient


def _build_tracing(settings: Settings) -> TracingClient:
    """Real Langfuse client when configured; otherwise a no-op (keeps local/offline boot working).

    The fail-loud tracing check lives in the separate ``python -m app.core.lifespan`` boot step.
    """
    if not (settings.tracing.enabled and settings.tracing.host):
        return NullTracingClient()
    try:
        return build_tracing_client(settings, VaultClient(settings.vault))
    except (TracingError, VaultError):
        return NullTracingClient()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    engine = create_engine(settings.database)
    # One shared model-server HTTP client for the process (no per-request httpx clients).
    model_server = ModelServerClient(settings.model_server)
    tracing = _build_tracing(settings)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.model_server = model_server
    app.state.tracing = tracing
    try:
        yield
    finally:
        tracing.flush()
        await model_server.aclose()
        await engine.dispose()
