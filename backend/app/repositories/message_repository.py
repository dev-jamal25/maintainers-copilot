"""SQL-only repository for ``messages`` (ordered turns within a conversation)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.chat import ChatMessage, MessageRole

_INSERT_SQL = text(
    """
    INSERT INTO messages (id, conversation_id, role, content, tool_name, created_at)
    VALUES (:id, :conversation_id, :role, :content, :tool_name, :created_at)
    """
)

_LIST_SQL = text(
    """
    SELECT id, conversation_id, role, content, tool_name, created_at
    FROM messages WHERE conversation_id = :conversation_id
    ORDER BY created_at, id
    """
)


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        tool_name: str | None = None,
    ) -> ChatMessage:
        now = datetime.now(UTC)
        message_id = uuid4()
        await self._session.execute(
            _INSERT_SQL,
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "role": role.value,
                "content": content,
                "tool_name": tool_name,
                "created_at": now,
            },
        )
        return ChatMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_name=tool_name,
            created_at=now,
        )

    async def list_for_conversation(self, conversation_id: UUID) -> list[ChatMessage]:
        result = await self._session.execute(_LIST_SQL, {"conversation_id": conversation_id})
        return [
            ChatMessage(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=MessageRole(row["role"]),
                content=str(row["content"]),
                tool_name=row["tool_name"],
                created_at=row["created_at"],
            )
            for row in result.mappings().all()
        ]
