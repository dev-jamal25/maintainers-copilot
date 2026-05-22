"""Maintainer's Copilot backend entrypoint.

``create_app()`` wires the lifespan, request-id middleware, the single domain-exception handler, and
the feature routers. Routes only coordinate HTTP concerns; business logic lives in services.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers, register_request_id_middleware
from app.api.health import router as health_router
from app.api.loader import router as loader_router
from app.api.memory import router as memory_router
from app.api.widget import router as widget_router
from app.core.app_lifespan import lifespan


def create_app() -> FastAPI:
    app = FastAPI(title="Maintainer's Copilot API", lifespan=lifespan)
    register_request_id_middleware(app)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(widget_router)
    app.include_router(memory_router)
    app.include_router(loader_router)
    return app


app = create_app()
