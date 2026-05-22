"""RagService: retrieval -> rerank -> parent-context -> grounded answer, plus degradation paths."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.exceptions import ExternalServiceUnavailable
from app.domain.rag import RetrievedChunk
from app.domain.tools import AnswerWithRagInput
from app.infra.errors import LLMUnavailableError, ModelServerUnavailableError
from app.infra.model_server_client import RerankHit
from app.repositories.chunk_repository import StoredChunk
from app.services.rag_service import RagParams, RagService


def _child(chunk_id: str, parent_id: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, parent_id=parent_id, score=score, rank=rank)


def _stored(chunk_id: str, parent_id: str | None, level: str, text: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        level=level,
        source_type="docs",
        title="Scheduler docs",
        url="https://docs/scheduler",
        text=text,
    )


class FakeChunkReader:
    def __init__(self, retrieved: list[RetrievedChunk], store: dict[str, StoredChunk]) -> None:
        self._retrieved = retrieved
        self._store = store
        self.search_kwargs: dict[str, object] = {}

    async def search_cosine(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        level: str = "child",
        source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        self.search_kwargs = {"limit": limit, "level": level, "source_type": source_type}
        return list(self._retrieved)

    async def fetch_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]:
        return {cid: self._store[cid] for cid in chunk_ids if cid in self._store}


class FakeModelServer:
    def __init__(self, *, rerank_hits: list[RerankHit] | None = None) -> None:
        self._rerank_hits = rerank_hits
        self.embed_calls = 0

    async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.embed_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankHit]:
        if self._rerank_hits is None:
            raise ModelServerUnavailableError("rerank down")
        return self._rerank_hits


class FakeLLM:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.last_prompt = ""

    async def generate(self, prompt: str, *, max_tokens: int) -> str:
        self.calls += 1
        self.last_prompt = prompt
        if self.fail:
            raise LLMUnavailableError("llm down")
        return "Grounded answer based on the context."


def _three_children_one_parent() -> tuple[list[RetrievedChunk], dict[str, StoredChunk]]:
    retrieved = [
        _child("c0", "p0", 0.9, 0),
        _child("c1", "p0", 0.8, 1),
        _child("c2", "p0", 0.7, 2),
    ]
    store = {
        "c0": _stored("c0", "p0", "child", "child zero text"),
        "c1": _stored("c1", "p0", "child", "child one text"),
        "c2": _stored("c2", "p0", "child", "child two text"),
        "p0": _stored("p0", None, "parent", "PARENT CONTEXT about the scheduler restart loop."),
    }
    return retrieved, store


async def test_happy_path_reranks_and_generates() -> None:
    retrieved, store = _three_children_one_parent()
    reader = FakeChunkReader(retrieved, store)
    model_server = FakeModelServer(rerank_hits=[RerankHit(2, 9.0), RerankHit(0, 1.0)])
    llm = FakeLLM()
    service = RagService(chunks=reader, model_server=model_server, llm=llm)

    result = await service.answer(AnswerWithRagInput(question="why does the scheduler restart?"))

    assert result.answer == "Grounded answer based on the context."
    assert "PARENT CONTEXT" in llm.last_prompt
    assert "why does the scheduler restart?" in llm.last_prompt
    # rerank put c2 first, then c0; both cite parent docs.
    assert [c.chunk_id for c in result.citations] == ["c2", "c0"]
    assert result.citations[0].source_type == "docs"


async def test_source_type_filter_is_passed_through() -> None:
    retrieved, store = _three_children_one_parent()
    reader = FakeChunkReader(retrieved, store)
    service = RagService(chunks=reader, model_server=FakeModelServer(rerank_hits=[]), llm=FakeLLM())
    await service.answer(AnswerWithRagInput(question="q", source_type="issue"))
    assert reader.search_kwargs["source_type"] == "issue"


async def test_no_chunks_returns_refusal_without_llm_call() -> None:
    reader = FakeChunkReader([], {})
    llm = FakeLLM()
    service = RagService(chunks=reader, model_server=FakeModelServer(rerank_hits=[]), llm=llm)
    result = await service.answer(AnswerWithRagInput(question="q"))
    assert "couldn't find relevant information" in result.answer
    assert result.citations == []
    assert llm.calls == 0


async def test_rerank_unavailable_falls_back_to_dense() -> None:
    retrieved, store = _three_children_one_parent()
    reader = FakeChunkReader(retrieved, store)
    model_server = FakeModelServer(rerank_hits=None)  # rerank raises
    llm = FakeLLM()
    service = RagService(chunks=reader, model_server=model_server, llm=llm)
    result = await service.answer(AnswerWithRagInput(question="q"))
    # Falls back to dense order c0, c1, c2.
    assert [c.chunk_id for c in result.citations] == ["c0", "c1", "c2"]
    assert llm.calls == 1


async def test_embed_unavailable_raises_external_service() -> None:
    class DownEmbed(FakeModelServer):
        async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
            raise ModelServerUnavailableError("embed down")

    reader = FakeChunkReader(*_three_children_one_parent())
    service = RagService(chunks=reader, model_server=DownEmbed(), llm=FakeLLM())
    with pytest.raises(ExternalServiceUnavailable):
        await service.answer(AnswerWithRagInput(question="q"))


async def test_llm_unavailable_raises_external_service() -> None:
    retrieved, store = _three_children_one_parent()
    reader = FakeChunkReader(retrieved, store)
    service = RagService(
        chunks=reader,
        model_server=FakeModelServer(rerank_hits=[RerankHit(0, 1.0)]),
        llm=FakeLLM(fail=True),
        params=RagParams(use_rerank=False),
    )
    with pytest.raises(ExternalServiceUnavailable):
        await service.answer(AnswerWithRagInput(question="q"))
