"""Memory inspector route: auth-gated, returns the current user's recent episodic memories."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import current_active_user, get_jwt_secret
from app.api.memory import get_memory_service
from app.db.models import User
from app.domain.memory import EpisodicEventType, EpisodicMemory
from app.main import create_app


class FakeMemoryService:
    def __init__(self, user_id: UUID) -> None:
        self._user_id = user_id

    async def list_recent(self, user_id: UUID, *, limit: int = 50) -> list[EpisodicMemory]:
        return [
            EpisodicMemory(
                id=uuid4(),
                user_id=user_id,
                event_type=EpisodicEventType.PREFERENCE,
                content="prefers concise replies",
                subject="style",
                created_at=datetime.now(UTC),
            )
        ]


def _app() -> FastAPI:
    app = create_app()
    user = User(
        id=uuid4(),
        email="dev@example.com",
        hashed_password="x",  # noqa: S106 - fake value for an offline test
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role="user",
    )
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    app.dependency_overrides[current_active_user] = lambda: user
    app.dependency_overrides[get_memory_service] = lambda: FakeMemoryService(user.id)
    return app


def test_list_memories_requires_auth() -> None:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    with TestClient(app) as client:
        assert client.get("/memories").status_code == 401


def test_list_memories_returns_user_memories() -> None:
    with TestClient(_app()) as client:
        response = client.get("/memories")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["event_type"] == "preference"
    assert body[0]["content"] == "prefers concise replies"
