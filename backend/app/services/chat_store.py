"""Conversation persistence for the chat route (service layer; owns the transaction).

Postgres is the durable store (conversations + messages, user-scoped). An optional Redis
``ConversationCache`` is a best-effort short-term layer: ``load_history`` prefers the cache and
falls back to Postgres (warming it); writes go to Postgres and are mirrored to the cache. Cache
errors never break chat — they fall back to the DB. Tool turns are not replayed (only user/assistant
text), which keeps multi-turn context simple and correct.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.chat import MessageRole
from app.domain.exceptions import NotFoundError
from app.infra.redis_cache import ConversationCache
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository

logger = logging.getLogger(__name__)

_TITLE_MAX = 80


class ChatStore:
    def __init__(self, session: AsyncSession, *, cache: ConversationCache | None = None) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
        self._cache = cache

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

    async def load_history(self, conversation_id: UUID, *, user_id: UUID) -> list[dict[str, Any]]:
        cached = await self._cache_load(user_id, conversation_id)
        if cached:
            return cached
        messages = await self._messages.list_for_conversation(conversation_id)
        history = [
            {"role": message.role.value, "content": message.content}
            for message in messages
            if message.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        if history:
            await self._cache_replace(user_id, conversation_id, history)
        return history

    async def add_user_message(self, conversation_id: UUID, content: str, *, user_id: UUID) -> None:
        await self._messages.add(
            conversation_id=conversation_id, role=MessageRole.USER, content=content
        )
        await self._cache_append(user_id, conversation_id, "user", content)

    async def add_assistant_message(
        self, conversation_id: UUID, content: str, *, user_id: UUID
    ) -> None:
        await self._messages.add(
            conversation_id=conversation_id, role=MessageRole.ASSISTANT, content=content
        )
        await self._conversations.touch(conversation_id)
        await self._cache_append(user_id, conversation_id, "assistant", content)

    async def commit(self) -> None:
        await self._session.commit()

    # --- best-effort short-term cache (never breaks chat) --------------------

    async def _cache_load(
        self, user_id: UUID, conversation_id: UUID
    ) -> list[dict[str, Any]] | None:
        if self._cache is None:
            return None
        try:
            return list(await self._cache.load(str(user_id), str(conversation_id)))
        except Exception:  # noqa: BLE001 - cache is best-effort; fall back to Postgres
            logger.warning("chat_cache_load_failed_falling_back_to_db")
            return None

    async def _cache_replace(
        self, user_id: UUID, conversation_id: UUID, history: list[dict[str, Any]]
    ) -> None:
        if self._cache is None:
            return
        try:
            messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in history]
            await self._cache.replace(str(user_id), str(conversation_id), messages)
        except Exception:  # noqa: BLE001 - best-effort
            logger.warning("chat_cache_warm_failed")

    async def _cache_append(
        self, user_id: UUID, conversation_id: UUID, role: str, content: str
    ) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.append(
                str(user_id), str(conversation_id), {"role": role, "content": content}
            )
        except Exception:  # noqa: BLE001 - best-effort
            logger.warning("chat_cache_append_failed")
