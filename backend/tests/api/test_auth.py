"""Auth contract tests (offline): the 401/403 boundary + require_admin role logic.

Full login + invite-accept flows touch Postgres and are exercised by the stack smoke path, not here;
these tests prove the authorization wiring without a live DB/Vault by overriding the secret + user
dependencies.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.auth import current_active_user, get_jwt_secret, require_admin
from app.db.models import User
from app.domain.exceptions import PermissionDenied
from app.main import create_app


def _user(role: str) -> User:
    return User(
        id=uuid4(),
        email=f"{role}@example.com",
        hashed_password="x",  # noqa: S106 - fake non-secret value for an offline unit test
        is_active=True,
        is_superuser=(role == "admin"),
        is_verified=True,
        role=role,
    )


def test_protected_route_returns_401_without_token() -> None:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    with TestClient(app) as client:
        response = client.get("/users/me")
    assert response.status_code == 401


def test_admin_route_returns_403_for_non_admin() -> None:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    app.dependency_overrides[current_active_user] = lambda: _user("user")
    with TestClient(app) as client:
        response = client.post(
            "/auth/invitations", json={"email": "new@example.com", "role": "user"}
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "permission_denied"


@pytest.mark.asyncio
async def test_require_admin_allows_admin() -> None:
    admin = _user("admin")
    assert await require_admin(admin) is admin


@pytest.mark.asyncio
async def test_require_admin_rejects_user_role() -> None:
    with pytest.raises(PermissionDenied):
        await require_admin(_user("user"))
