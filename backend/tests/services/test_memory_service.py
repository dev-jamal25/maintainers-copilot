"""MemoryService: redaction before persistence, an audit row per write, user-isolated recall."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.memory import (
    EpisodicEventType,
    EpisodicMemory,
    MemoryWriteRequest,
    RetrievedMemory,
)
from app.services.memory_service import MemoryService


class CapturingStore:
    def __init__(self) -> None:
        self.inserted: dict[str, object] = {}
        self.search_user: UUID | None = None

    async def insert(
        self,
        *,
        user_id: UUID,
        event_type: EpisodicEventType,
        content: str,
        subject: str | None,
        embedding: Sequence[float] | None,
        created_at: datetime,
    ) -> UUID:
        memory_id = uuid4()
        self.inserted = {
            "user_id": user_id,
            "content": content,
            "subject": subject,
            "embedding": embedding,
        }
        return memory_id

    async def search_cosine(
        self, user_id: UUID, query_embedding: Sequence[float], *, limit: int
    ) -> list[RetrievedMemory]:
        self.search_user = user_id
        return [
            RetrievedMemory(
                id=uuid4(),
                event_type=EpisodicEventType.PREFERENCE,
                content="prefers concise replies",
                subject="style",
                score=0.9,
                created_at=datetime.now(UTC),
            )
        ]

    async def list_recent(self, user_id: UUID, *, limit: int = 50) -> list[EpisodicMemory]:
        return []


class FakeEmbedder:
    async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class CapturingAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def record(
        self,
        *,
        action: str,
        actor_id: UUID | None,
        target_type: str | None,
        target_id: str | None,
        request_id: str | None,
        trace_id: str | None,
        details: dict[str, object] | None,
        created_at: datetime,
    ) -> UUID:
        self.records.append({"action": action, "actor_id": actor_id, "details": details})
        return uuid4()


def _service(store: CapturingStore, audit: CapturingAudit) -> MemoryService:
    return MemoryService(memories=store, embedder=FakeEmbedder(), audit=audit)


async def test_write_redacts_content_before_persisting() -> None:
    store, audit = CapturingStore(), CapturingAudit()
    user_id = uuid4()
    request = MemoryWriteRequest(
        event_type=EpisodicEventType.FACT,
        content="deploy key is sk-ant-api03-LEAKED000111222333444555666",
        subject="contact alice@example.com",
    )
    await _service(store, audit).write_memory(user_id, request, request_id="rid-1")

    assert "sk-ant-api03-LEAKED" not in str(store.inserted["content"])
    assert "alice@example.com" not in str(store.inserted["subject"])
    assert store.inserted["user_id"] == user_id
    assert store.inserted["embedding"] is not None


async def test_write_creates_one_audit_row() -> None:
    store, audit = CapturingStore(), CapturingAudit()
    user_id = uuid4()
    await _service(store, audit).write_memory(
        user_id,
        MemoryWriteRequest(event_type=EpisodicEventType.SAVED_SUMMARY, content="saved summary X"),
        request_id="rid-2",
    )
    assert len(audit.records) == 1
    assert audit.records[0]["action"] == "memory.write"
    assert audit.records[0]["actor_id"] == user_id


async def test_recall_is_scoped_to_the_requesting_user() -> None:
    store, audit = CapturingStore(), CapturingAudit()
    user_id = uuid4()
    results = await _service(store, audit).recall(user_id, "what style do I prefer?")
    assert store.search_user == user_id  # repository filtered by this user
    assert results and results[0].subject == "style"
