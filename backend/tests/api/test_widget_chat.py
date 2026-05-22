"""Anonymous widget chat: origin-gated, ephemeral Redis session, no auth, restricted tools.

Heavy deps (chat loop, Redis, widget repo) are overridden so the endpoint is verifiable offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import get_chat_service_factory
from app.api.widget import get_conversation_cache, get_widget_service
from app.domain.chat import SSEEventType
from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPosition, WidgetTheme
from app.main import create_app
from app.services.chat_service import ChatEvent, ChatResult
from app.services.widget_service import WidgetService

ALLOWED = "https://allowed.example"
BLOCKED = "https://evil.example"


def _config(active: bool = True) -> WidgetConfig:
    return WidgetConfig(
        widget_id="w_demo",
        allowed_origins=[ALLOWED],
        theme=WidgetTheme.LIGHT,
        primary_color="#2563eb",
        position=WidgetPosition.BOTTOM_RIGHT,
        greeting="Hi",
        enabled_tools=["answer_with_rag", "write_memory"],  # write_memory is stripped for anon
        is_active=active,
        created_by=uuid4(),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class FakeWidgetRepo:
    def __init__(self, *, active: bool = True) -> None:
        self._active = active

    async def create(
        self, *, widget_id: str, payload: WidgetCreate, created_by: Any
    ) -> WidgetConfig:
        return _config()

    async def get(self, widget_id: str) -> WidgetConfig | None:
        return _config(self._active) if widget_id == "w_demo" else None

    async def list_all(self) -> list[WidgetConfig]:
        return [_config()]


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, list[dict[str, str]]] = {}

    async def load(self, user_id: str, conversation_id: str) -> list[dict[str, str]]:
        return self.store.get(f"{user_id}:{conversation_id}", [])

    async def append(self, user_id: str, conversation_id: str, message: dict[str, str]) -> None:
        self.store.setdefault(f"{user_id}:{conversation_id}", []).append(message)


class FakeChatService:
    def __init__(self) -> None:
        self.allowed_seen = True

    async def run_turn(self, **kwargs: Any) -> ChatResult:
        return ChatResult(
            answer="Grounded answer.",
            events=[
                ChatEvent(SSEEventType.TOOL_CALL, {"tool": "answer_with_rag"}),
                ChatEvent(SSEEventType.DONE, {"text": "Grounded answer."}),
            ],
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(SSEEventType.TOOL_CALL, {"tool": "answer_with_rag"})
        yield ChatEvent(SSEEventType.DONE, {"text": "Grounded answer."})


def _app(*, active: bool = True) -> tuple[FastAPI, FakeCache]:
    app = create_app()
    cache = FakeCache()
    app.dependency_overrides[get_widget_service] = lambda: WidgetService(
        FakeWidgetRepo(active=active)
    )
    app.dependency_overrides[get_conversation_cache] = lambda: cache
    app.dependency_overrides[get_chat_service_factory] = lambda: lambda allowed: FakeChatService()
    return app, cache


def test_blocked_origin_is_forbidden() -> None:
    app, _ = _app()
    with TestClient(app) as client:
        response = client.post(
            "/widgets/w_demo/chat", json={"message": "hi"}, headers={"Origin": BLOCKED}
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "permission_denied"


def test_missing_origin_is_forbidden() -> None:
    app, _ = _app()
    with TestClient(app) as client:
        response = client.post("/widgets/w_demo/chat", json={"message": "hi"})
    assert response.status_code == 403


def test_allowed_origin_chats_and_caches() -> None:
    app, cache = _app()
    with TestClient(app) as client:
        response = client.post(
            "/widgets/w_demo/chat", json={"message": "why?"}, headers={"Origin": ALLOWED}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer."
    assert body["session_id"]
    assert response.headers.get("access-control-allow-origin") == ALLOWED
    # user + assistant persisted to the ephemeral Redis session.
    key = f"widget:w_demo:{body['session_id']}"
    assert len(cache.store[key]) == 2


def test_inactive_widget_is_forbidden() -> None:
    app, _ = _app(active=False)
    with TestClient(app) as client:
        response = client.post(
            "/widgets/w_demo/chat", json={"message": "hi"}, headers={"Origin": ALLOWED}
        )
    assert response.status_code == 403


def test_missing_widget_is_404() -> None:
    app, _ = _app()
    with TestClient(app) as client:
        response = client.post(
            "/widgets/nope/chat", json={"message": "hi"}, headers={"Origin": ALLOWED}
        )
    assert response.status_code == 404


def test_stream_emits_named_sse_events() -> None:
    app, cache = _app()
    with TestClient(app) as client:
        response = client.post(
            "/widgets/w_demo/chat/stream", json={"message": "hi"}, headers={"Origin": ALLOWED}
        )
    assert response.status_code == 200
    text = response.text
    assert "event: session" in text
    assert "event: done" in text
