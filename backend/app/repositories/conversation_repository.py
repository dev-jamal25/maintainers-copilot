"""SQL-only repository for ``conversations``. Reads are user-scoped (no cross-user access)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.chat import Conversation

_INSERT_SQL = text(
    """
    INSERT INTO conversations (id, user_id, title, created_at, updated_at)
    VALUES (:id, :user_id, :title, :created_at, :created_at)
    """
)

_GET_SQL = text(
    """
    SELECT id, user_id, title, created_at, updated_at
    FROM conversations WHERE id = :id AND user_id = :user_id
    """
)

_TOUCH_SQL = text("UPDATE conversations SET updated_at = :now WHERE id = :id")


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, user_id: UUID, title: str | None) -> Conversation:
        now = datetime.now(UTC)
        conversation_id = uuid4()
        await self._session.execute(
            _INSERT_SQL,
            {"id": conversation_id, "user_id": user_id, "title": title, "created_at": now},
        )
        return Conversation(
            id=conversation_id, user_id=user_id, title=title, created_at=now, updated_at=now
        )

    async def get(self, conversation_id: UUID, *, user_id: UUID) -> Conversation | None:
        result = await self._session.execute(_GET_SQL, {"id": conversation_id, "user_id": user_id})
        row = result.mappings().first()
        if row is None:
            return None
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def touch(self, conversation_id: UUID) -> None:
        await self._session.execute(_TOUCH_SQL, {"id": conversation_id, "now": datetime.now(UTC)})
