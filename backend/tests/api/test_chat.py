"""Chat route + SSE: auth boundary, persistence ordering, and named SSE events.

The heavy components (LLM, tools, DB) are overridden so the HTTP/SSE + auth layer is verifiable
offline. ChatService and ChatStore behaviour have their own unit tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.api.auth import current_active_user, get_jwt_secret
from app.api.chat import get_chat_service, get_chat_store
from app.db.models import User
from app.domain.chat import SSEEventType
from app.main import create_app
from app.services.chat_service import ChatEvent, ChatResult


def _user() -> User:
    return User(
        id=uuid4(),
        email="dev@example.com",
        hashed_password="x",  # noqa: S106 - fake value for an offline test
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role="user",
    )


class FakeChatService:
    async def run_turn(self, **kwargs: Any) -> ChatResult:
        return ChatResult(
            answer="Final answer.",
            events=[
                ChatEvent(SSEEventType.TOOL_CALL, {"tool": "answer_with_rag"}),
                ChatEvent(SSEEventType.TOOL_RESULT, {"tool": "answer_with_rag", "is_error": False}),
                ChatEvent(SSEEventType.DONE, {"text": "Final answer."}),
            ],
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(SSEEventType.TOOL_CALL, {"tool": "answer_with_rag"})
        yield ChatEvent(SSEEventType.DONE, {"text": "Final answer."})


class FakeChatStore:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []
        self.committed = False

    async def ensure_conversation(
        self, conversation_id: UUID | None, *, user_id: UUID, first_message: str
    ) -> UUID:
        return conversation_id or uuid4()

    async def load_history(self, conversation_id: UUID, *, user_id: UUID) -> list[dict[str, Any]]:
        return []

    async def add_user_message(self, conversation_id: UUID, content: str, *, user_id: UUID) -> None:
        self.user_messages.append(content)

    async def add_assistant_message(
        self, conversation_id: UUID, content: str, *, user_id: UUID
    ) -> None:
        self.assistant_messages.append(content)

    async def commit(self) -> None:
        self.committed = True


def _client(store: FakeChatStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    app.dependency_overrides[current_active_user] = _user
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    app.dependency_overrides[get_chat_store] = lambda: store
    return TestClient(app)


def test_chat_requires_authentication() -> None:
    app = create_app()
    app.dependency_overrides[get_jwt_secret] = lambda: "test-secret"
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401


def test_chat_returns_answer_and_persists_in_order() -> None:
    store = FakeChatStore()
    with _client(store) as client:
        response = client.post("/chat", json={"message": "why does the scheduler restart?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Final answer."
    assert any(e["type"] == "tool_call" for e in body["events"])
    assert store.user_messages == ["why does the scheduler restart?"]
    assert store.assistant_messages == ["Final answer."]
    assert store.committed is True


def test_chat_stream_emits_named_sse_events() -> None:
    store = FakeChatStore()
    with _client(store) as client:
        response = client.post("/chat/stream", json={"message": "hi"})
    assert response.status_code == 200
    text = response.text
    assert "event: tool_call" in text
    assert "event: done" in text
    assert store.assistant_messages == ["Final answer."]  # persisted after stream
