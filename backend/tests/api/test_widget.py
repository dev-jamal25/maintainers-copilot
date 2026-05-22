"""Widget API: public config hides origins, CORS per allowed origins, CSP on embed, admin gate."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import current_active_user, get_jwt_secret, require_admin
from app.api.deps import get_session
from app.api.widget import get_widget_service
from app.db.models import User
from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPosition, WidgetTheme
from app.main import create_app
from app.services.widget_service import (
    WidgetService,
    csp_frame_ancestors,
    is_origin_allowed,
)

ALLOWED = "https://allowed.example"
BLOCKED = "https://evil.example"


def _config() -> WidgetConfig:
    return WidgetConfig(
        widget_id="w_demo",
        allowed_origins=[ALLOWED],
        theme=WidgetTheme.DARK,
        primary_color="#123456",
        position=WidgetPosition.BOTTOM_LEFT,
        greeting="Hi there",
        enabled_tools=["answer_with_rag"],
        is_active=True,
        created_by=uuid4(),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class FakeSession:
    async def commit(self) -> None:
        return None


class FakeWidgetRepo:
    def __init__(self) -> None:
        self.created: WidgetConfig | None = None

    async def create(
        self, *, widget_id: str, payload: WidgetCreate, created_by: UUID
    ) -> WidgetConfig:
        self.created = _config()
        return self.created

    async def get(self, widget_id: str) -> WidgetConfig | None:
        return _config() if widget_id == "w_demo" else None

    async def list_all(self) -> list[WidgetConfig]:
        return [_config()]


def _client() -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    app.dependency_overrides[get_widget_service] = lambda: WidgetService(FakeWidgetRepo())
    return app


# --- pure helpers ------------------------------------------------------------


def test_is_origin_allowed_exact_match_only() -> None:
    assert is_origin_allowed([ALLOWED], ALLOWED) is True
    assert is_origin_allowed([ALLOWED], BLOCKED) is False
    assert is_origin_allowed([ALLOWED], None) is False
    assert is_origin_allowed([], ALLOWED) is False


def test_csp_frame_ancestors() -> None:
    assert csp_frame_ancestors([ALLOWED]) == f"frame-ancestors {ALLOWED}"
    assert csp_frame_ancestors([]) == "frame-ancestors 'none'"


# --- routes ------------------------------------------------------------------


def test_public_config_hides_allowed_origins() -> None:
    with TestClient(_client()) as client:
        response = client.get("/widgets/w_demo/config")
    assert response.status_code == 200
    body = response.json()
    assert "allowed_origins" not in body
    assert "created_by" not in body
    assert body["widget_id"] == "w_demo"
    assert body["primary_color"] == "#123456"


def test_config_sets_cors_only_for_allowed_origin() -> None:
    with TestClient(_client()) as client:
        allowed = client.get("/widgets/w_demo/config", headers={"Origin": ALLOWED})
        blocked = client.get("/widgets/w_demo/config", headers={"Origin": BLOCKED})
    assert allowed.headers.get("access-control-allow-origin") == ALLOWED
    assert blocked.headers.get("access-control-allow-origin") is None


def test_missing_widget_returns_404() -> None:
    with TestClient(_client()) as client:
        response = client.get("/widgets/nope/config")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_embed_sets_csp_frame_ancestors() -> None:
    with TestClient(_client()) as client:
        response = client.get("/widgets/w_demo/embed")
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == f"frame-ancestors {ALLOWED}"
    assert "w_demo" in response.text


def test_create_widget_requires_admin() -> None:
    app = _client()
    app.dependency_overrides[current_active_user] = lambda: User(
        id=uuid4(),
        email="u@example.com",
        hashed_password="x",  # noqa: S106 - fake value for an offline test
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role="user",
    )
    with TestClient(app) as client:
        response = client.post("/widgets", json={"allowed_origins": [ALLOWED]})
    assert response.status_code == 403


def test_create_widget_succeeds_for_admin() -> None:
    app = _client()
    admin = User(
        id=uuid4(),
        email="admin@example.com",
        hashed_password="x",  # noqa: S106 - fake value for an offline test
        is_active=True,
        is_superuser=True,
        is_verified=True,
        role="admin",
    )
    app.dependency_overrides[require_admin] = lambda: admin
    app.dependency_overrides[get_session] = lambda: FakeSession()
    with TestClient(app) as client:
        response = client.post("/widgets", json={"allowed_origins": [ALLOWED], "theme": "dark"})
    assert response.status_code == 201
    assert response.json()["widget_id"] == "w_demo"
