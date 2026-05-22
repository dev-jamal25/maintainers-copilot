"""ConversationCache: per-(user,conversation) key isolation, TTL on every write, round-trip."""

from __future__ import annotations

from app.infra.redis_cache import ConversationCache


class FakeRedis:
    """In-memory stand-in implementing the KeyValueClient surface, recording the last TTL."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.last_ex: int | None = None

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> object:
        self.store[key] = value
        self.last_ex = ex
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += 1 if self.store.pop(key, None) is not None else 0
        return removed


async def test_append_and_load_round_trip_with_ttl() -> None:
    client = FakeRedis()
    cache = ConversationCache(client, ttl_seconds=86_400)
    await cache.append("user-1", "conv-1", {"role": "user", "content": "hi"})
    await cache.append("user-1", "conv-1", {"role": "assistant", "content": "hello"})

    messages = await cache.load("user-1", "conv-1")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert client.last_ex == 86_400  # TTL set on write
    assert "chat:session:user-1:conv-1" in client.store


async def test_users_are_isolated_even_with_same_conversation_id() -> None:
    client = FakeRedis()
    cache = ConversationCache(client, ttl_seconds=60)
    await cache.append("user-A", "conv-X", {"role": "user", "content": "secret A"})
    await cache.append("user-B", "conv-X", {"role": "user", "content": "secret B"})

    a_messages = await cache.load("user-A", "conv-X")
    b_messages = await cache.load("user-B", "conv-X")
    assert a_messages[0]["content"] == "secret A"
    assert b_messages[0]["content"] == "secret B"
    assert len(a_messages) == 1 and len(b_messages) == 1  # no cross-user bleed


async def test_clear_removes_only_that_session() -> None:
    client = FakeRedis()
    cache = ConversationCache(client, ttl_seconds=60)
    await cache.append("user-1", "conv-1", {"role": "user", "content": "x"})
    await cache.append("user-1", "conv-2", {"role": "user", "content": "y"})
    await cache.clear("user-1", "conv-1")
    assert await cache.load("user-1", "conv-1") == []
    assert await cache.load("user-1", "conv-2") != []
