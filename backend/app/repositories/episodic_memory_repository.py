"""SQL-only repository for ``episodic_memories`` (long-term episodic memory, pgvector).

User isolation is enforced in SQL: every read filters ``user_id`` so recall can never cross users.
Embeddings use the shared pgvector text encoding (see ``chunk_repository.encode_vector``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.memory import EpisodicEventType, EpisodicMemory, RetrievedMemory
from app.repositories.chunk_repository import encode_vector

_INSERT_SQL = text(
    """
    INSERT INTO episodic_memories (
        id, user_id, event_type, content, subject, embedding, created_at
    ) VALUES (
        :id, :user_id, :event_type, :content, :subject, CAST(:embedding AS vector), :created_at
    )
    """
)

_SEARCH_SQL = text(
    """
    SELECT id, event_type, content, subject, created_at,
           (1 - (embedding <=> CAST(:query AS vector))) AS score
    FROM episodic_memories
    WHERE user_id = :user_id AND embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :limit
    """
)

_LIST_RECENT_SQL = text(
    """
    SELECT id, event_type, content, subject, created_at
    FROM episodic_memories
    WHERE user_id = :user_id
    ORDER BY created_at DESC
    LIMIT :limit
    """
)


class EpisodicMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        await self._session.execute(
            _INSERT_SQL,
            {
                "id": memory_id,
                "user_id": user_id,
                "event_type": event_type.value,
                "content": content,
                "subject": subject,
                "embedding": encode_vector(embedding) if embedding is not None else None,
                "created_at": created_at,
            },
        )
        return memory_id

    async def search_cosine(
        self, user_id: UUID, query_embedding: Sequence[float], *, limit: int
    ) -> list[RetrievedMemory]:
        result = await self._session.execute(
            _SEARCH_SQL,
            {"user_id": user_id, "query": encode_vector(query_embedding), "limit": limit},
        )
        return [
            RetrievedMemory(
                id=row["id"],
                event_type=EpisodicEventType(row["event_type"]),
                content=str(row["content"]),
                subject=row["subject"],
                score=float(row["score"]),
                created_at=row["created_at"],
            )
            for row in result.mappings().all()
        ]

    async def list_recent(self, user_id: UUID, *, limit: int = 50) -> list[EpisodicMemory]:
        result = await self._session.execute(
            _LIST_RECENT_SQL, {"user_id": user_id, "limit": limit}
        )
        return [
            EpisodicMemory(
                id=row["id"],
                user_id=user_id,
                event_type=EpisodicEventType(row["event_type"]),
                content=str(row["content"]),
                subject=row["subject"],
                created_at=row["created_at"],
            )
            for row in result.mappings().all()
        ]
