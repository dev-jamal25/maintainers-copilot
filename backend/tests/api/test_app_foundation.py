"""FastAPI foundation: health, request-id propagation, and the AppError->HTTP mapping.

Verifies users never see stack traces or internal detail, and that every response carries an
X-Request-ID. Uses a throwaway app with deliberately-failing routes to exercise the handlers.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import (
    REQUEST_ID_HEADER,
    register_exception_handlers,
    register_request_id_middleware,
)
from app.domain.exceptions import NotFoundError
from app.main import create_app


def test_health_ok_and_has_request_id() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get(REQUEST_ID_HEADER)


def _failing_app() -> FastAPI:
    app = FastAPI()
    register_request_id_middleware(app)
    register_exception_handlers(app)

    class Body(BaseModel):
        n: int

    @app.get("/notfound")
    async def notfound() -> None:
        raise NotFoundError("conversation 7 not found", detail="internal-secret-detail")

    @app.get("/crash")
    async def crash() -> None:
        raise RuntimeError("boom leaking sk-ant-shouldnotappear")

    @app.post("/validate")
    async def validate(body: Body) -> dict[str, int]:
        return {"n": body.n}

    return app


def test_app_error_maps_to_status_and_hides_detail() -> None:
    client = TestClient(_failing_app(), raise_server_exceptions=False)
    response = client.get("/notfound")
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "conversation 7 not found"}
    }
    assert "internal-secret-detail" not in response.text
    assert response.headers.get(REQUEST_ID_HEADER)


def test_unhandled_exception_returns_safe_500() -> None:
    client = TestClient(_failing_app(), raise_server_exceptions=False)
    response = client.get("/crash")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Internal server error"}
    }
    # Neither the exception message nor any leaked secret appears in the body.
    assert "boom" not in response.text
    assert "sk-ant-shouldnotappear" not in response.text


def test_validation_error_shape() -> None:
    client = TestClient(_failing_app(), raise_server_exceptions=False)
    response = client.post("/validate", json={"n": "not-an-int"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["fields"], list)


def test_incoming_request_id_is_echoed() -> None:
    client = TestClient(create_app())
    response = client.get("/health", headers={REQUEST_ID_HEADER: "rid-12345"})
    assert response.headers.get(REQUEST_ID_HEADER) == "rid-12345"
