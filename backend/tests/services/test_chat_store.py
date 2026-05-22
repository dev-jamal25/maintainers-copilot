"""ChatStore short-term cache: cache hit skips DB, writes mirror to cache, errors fall back."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.redis_cache import ConversationCache
from app.services.chat_store import ChatStore


class _FakeResult:
    def mappings(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return []


class FakeSession:
    def __init__(self) -> None:
        self.executed = False

    async def execute(self, *args: Any, **kwargs: Any) -> _FakeResult:
        self.executed = True
        return _FakeResult()

    async def commit(self) -> None:
        return None


class FakeCache:
    def __init__(self, data: list[dict[str, str]] | None = None, *, fail: bool = False) -> None:
        self._data = data or []
        self.appended: list[dict[str, str]] = []
        self._fail = fail

    async def load(self, user_id: str, conversation_id: str) -> list[dict[str, str]]:
        if self._fail:
            raise RuntimeError("redis down")
        return list(self._data)

    async def replace(
        self, user_id: str, conversation_id: str, messages: list[dict[str, str]]
    ) -> None:
        return None

    async def append(self, user_id: str, conversation_id: str, message: dict[str, str]) -> None:
        if self._fail:
            raise RuntimeError("redis down")
        self.appended.append(message)


def _store(session: FakeSession, cache: FakeCache) -> ChatStore:
    return ChatStore(cast(AsyncSession, session), cache=cast(ConversationCache, cache))


async def test_cache_hit_skips_the_database() -> None:
    session = FakeSession()
    cache = FakeCache([{"role": "user", "content": "hi"}])
    history = await _store(session, cache).load_history(uuid4(), user_id=uuid4())
    assert history == [{"role": "user", "content": "hi"}]
    assert session.executed is False  # DB not touched on a cache hit


async def test_cache_miss_falls_back_to_db() -> None:
    session = FakeSession()
    history = await _store(session, FakeCache([])).load_history(uuid4(), user_id=uuid4())
    assert history == []
    assert session.executed is True  # fell through to Postgres


async def test_writes_mirror_to_cache() -> None:
    cache = FakeCache()
    store = _store(FakeSession(), cache)
    await store.add_user_message(uuid4(), "hello", user_id=uuid4())
    await store.add_assistant_message(uuid4(), "hi there", user_id=uuid4())
    assert cache.appended == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


async def test_cache_errors_never_break_chat() -> None:
    session = FakeSession()
    store = _store(session, FakeCache(fail=True))
    # load falls back to DB despite the cache raising; writes do not raise.
    assert await store.load_history(uuid4(), user_id=uuid4()) == []
    await store.add_user_message(uuid4(), "hello", user_id=uuid4())  # must not raise
