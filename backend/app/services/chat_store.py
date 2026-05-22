"""Conversation persistence for the chat route (service layer; owns the transaction).

Wraps the conversation + message repositories: ensure/lookup a conversation (user-scoped), load
prior turns as LLM-ready messages, append user/assistant messages, commit. Tool turns are not
replayed to the model (only user/assistant text), which keeps multi-turn context simple and correct.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.chat import MessageRole
from app.domain.exceptions import NotFoundError
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository

_TITLE_MAX = 80


class ChatStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)

    async def ensure_conversation(
        self, conversation_id: UUID | None, *, user_id: UUID, first_message: str
    ) -> UUID:
        if conversation_id is None:
            conversation = await self._conversations.create(
                user_id=user_id, title=first_message[:_TITLE_MAX]
            )
            return conversation.id
        existing = await self._conversations.get(conversation_id, user_id=user_id)
        if existing is None:
            raise NotFoundError("Conversation not found.")
        return existing.id

    async def load_history(self, conversation_id: UUID) -> list[dict[str, Any]]:
        messages = await self._messages.list_for_conversation(conversation_id)
        return [
            {"role": message.role.value, "content": message.content}
            for message in messages
            if message.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]

    async def add_user_message(self, conversation_id: UUID, content: str) -> None:
        await self._messages.add(
            conversation_id=conversation_id, role=MessageRole.USER, content=content
        )

    async def add_assistant_message(self, conversation_id: UUID, content: str) -> None:
        await self._messages.add(
            conversation_id=conversation_id, role=MessageRole.ASSISTANT, content=content
        )
        await self._conversations.touch(conversation_id)

    async def commit(self) -> None:
        await self._session.commit()
