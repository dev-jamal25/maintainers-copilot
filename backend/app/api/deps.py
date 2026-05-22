"""Shared FastAPI dependencies (DI seams).

Routes resolve settings, a DB session, and the request id through these — never reaching into
``app.state`` directly. Resources are built in the lifespan and read here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.infra.model_server_client import ModelServerClient
from app.infra.tracing import TracingClient


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_model_server(request: Request) -> ModelServerClient:
    """The process-wide model-server client built in the lifespan."""
    return cast(ModelServerClient, request.app.state.model_server)


def get_tracing(request: Request) -> TracingClient:
    """The process-wide tracing client built in the lifespan (Null when not configured)."""
    return cast(TracingClient, request.app.state.tracing)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = cast("async_sessionmaker[AsyncSession]", request.app.state.session_factory)
    async with factory() as session:
        yield session


def get_request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
ModelServerDep = Annotated[ModelServerClient, Depends(get_model_server)]
TracingDep = Annotated[TracingClient, Depends(get_tracing)]
