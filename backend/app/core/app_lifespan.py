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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    engine = create_engine(settings.database)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    try:
        yield
    finally:
        await engine.dispose()
