"""Long-term episodic memory service (CLAUDE.md memory: long-term, episodic).

Writes happen ONLY here (the explicit ``write_memory`` tool calls this — no automatic writes). Each
write redacts content + subject BEFORE persistence, embeds via the model-server (best-effort: a
memory is still stored if embedding is unavailable, just without a vector), and records an audit row
with redacted details. Recall is user-isolated (the repository filters ``user_id``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from app.domain.exceptions import ExternalServiceUnavailable
from app.domain.memory import (
    EpisodicEventType,
    EpisodicMemory,
    MemoryWriteRequest,
    RetrievedMemory,
)
from app.infra.errors import ModelServerError
from app.infra.redaction import redact, redact_value

logger = logging.getLogger(__name__)


class EmbedderPort(Protocol):
    async def embed(self, texts: Sequence[str], *, is_query: bool = ...) -> list[list[float]]: ...


class EpisodicStorePort(Protocol):
    async def insert(
        self,
        *,
        user_id: UUID,
        event_type: EpisodicEventType,
        content: str,
        subject: str | None,
        embedding: Sequence[float] | None,
        created_at: datetime,
    ) -> UUID: ...

    async def search_cosine(
        self, user_id: UUID, query_embedding: Sequence[float], *, limit: int
    ) -> list[RetrievedMemory]: ...

    async def list_recent(self, user_id: UUID, *, limit: int = ...) -> list[EpisodicMemory]: ...


class AuditPort(Protocol):
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
    ) -> UUID: ...


class MemoryService:
    def __init__(
        self,
        *,
        memories: EpisodicStorePort,
        embedder: EmbedderPort,
        audit: AuditPort,
        recall_limit: int = 5,
    ) -> None:
        self._memories = memories
        self._embedder = embedder
        self._audit = audit
        self._recall_limit = recall_limit

    async def write_memory(
        self,
        user_id: UUID,
        request: MemoryWriteRequest,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> EpisodicMemory:
        now = datetime.now(UTC)
        content = redact(request.content)
        subject = redact(request.subject) if request.subject else None

        embedding: list[float] | None = None
        try:
            vectors = await self._embedder.embed([content], is_query=False)
            embedding = vectors[0] if vectors else None
        except ModelServerError:
            logger.warning("memory_embed_unavailable_storing_without_vector")

        memory_id = await self._memories.insert(
            user_id=user_id,
            event_type=request.event_type,
            content=content,
            subject=subject,
            embedding=embedding,
            created_at=now,
        )
        await self._audit.record(
            action="memory.write",
            actor_id=user_id,
            target_type="episodic_memory",
            target_id=str(memory_id),
            request_id=request_id,
            trace_id=trace_id,
            details=redact_value({"event_type": request.event_type.value, "subject": subject}),
            created_at=now,
        )
        return EpisodicMemory(
            id=memory_id,
            user_id=user_id,
            event_type=request.event_type,
            content=content,
            subject=subject,
            created_at=now,
        )

    async def recall(
        self, user_id: UUID, query: str, *, limit: int | None = None
    ) -> list[RetrievedMemory]:
        try:
            vectors = await self._embedder.embed([query], is_query=True)
        except ModelServerError as exc:
            raise ExternalServiceUnavailable("The embedding service is unavailable.") from exc
        if not vectors or not vectors[0]:
            return []
        return await self._memories.search_cosine(
            user_id, vectors[0], limit=limit or self._recall_limit
        )

    async def list_recent(self, user_id: UUID, *, limit: int = 50) -> list[EpisodicMemory]:
        """Return the user's most recent episodic memories (for the memory inspector)."""
        return await self._memories.list_recent(user_id, limit=limit)
