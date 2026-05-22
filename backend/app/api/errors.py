"""API-layer error handling: request-id propagation + the single AppError->HTTP mapping.

CLAUDE.md: a single handler maps domain exceptions to structured responses; users never see stack
traces; never return 200 with an error body. Internal ``detail`` is logged (after redaction), never
returned. Every response carries ``X-Request-ID`` for log/trace correlation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.exceptions import AppError
from app.infra.redaction import redact

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"


def register_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
        request.state.request_id = request_id
        # One trace per request; the chat/tool spans use this as their trace id.
        request.state.trace_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, AppError)
        request_id = _request_id(request)
        logger.warning(
            "app_error",
            extra={
                "code": exc.code,
                "status_code": exc.status_code,
                "request_id": request_id,
                "trace_id": request_id,
                "detail": redact(exc.detail) if exc.detail else None,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        # Surface field locations + messages only — never the raw input values (may carry secrets).
        fields = [
            {"loc": list(error.get("loc", [])), "msg": str(error.get("msg", ""))}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "fields": fields,
                }
            },
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception(
            "unhandled_exception",
            extra={
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id,
                "trace_id": request_id,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
            headers={REQUEST_ID_HEADER: request_id},
        )
