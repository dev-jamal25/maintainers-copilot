"""Short-term conversation state in Redis (CLAUDE.md memory: short-term).

Keys are namespaced ``chat:session:{user_id}:{conversation_id}`` so state is strictly isolated per
user + conversation — there is no path by which one user reads another's session. Every write resets
the TTL (default 24h, ``settings.redis.short_term_ttl_seconds``) so abandoned sessions self-evict.
The KV client is injected behind a small Protocol so this unit-tests without a live Redis.
"""

from __future__ import annotations

import json
from typing import Protocol, cast


class KeyValueClient(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, *, ex: int | None = None) -> object: ...

    async def delete(self, *keys: str) -> int: ...


def build_redis_client(url: str) -> KeyValueClient:  # pragma: no cover - needs live Redis
    import redis.asyncio as redis

    return cast(KeyValueClient, redis.from_url(url, decode_responses=True))


class ConversationCache:
    """Stores the active message list for one (user, conversation) as a JSON blob with TTL."""

    def __init__(self, client: KeyValueClient, *, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(user_id: str, conversation_id: str) -> str:
        return f"chat:session:{user_id}:{conversation_id}"

    async def load(self, user_id: str, conversation_id: str) -> list[dict[str, str]]:
        raw = await self._client.get(self._key(user_id, conversation_id))
        if not raw:
            return []
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []

    async def replace(
        self, user_id: str, conversation_id: str, messages: list[dict[str, str]]
    ) -> None:
        await self._client.set(
            self._key(user_id, conversation_id),
            json.dumps(messages),
            ex=self._ttl_seconds,
        )

    async def append(self, user_id: str, conversation_id: str, message: dict[str, str]) -> None:
        messages = await self.load(user_id, conversation_id)
        messages.append(message)
        await self.replace(user_id, conversation_id, messages)

    async def clear(self, user_id: str, conversation_id: str) -> None:
        await self._client.delete(self._key(user_id, conversation_id))
