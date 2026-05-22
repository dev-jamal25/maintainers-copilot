"""Runtime RAG orchestration (separate from the Day-3 eval harness).

Flow: embed the query (model-server, query-prefixed) -> pgvector cosine retrieval over child chunks
(optional source_type metadata filter) -> optional cross-encoder rerank (model-server) -> expand
chosen children to their parent chunks for grounding context -> generate a grounded answer (LLM) ->
return answer + citations. Knobs (dense_top_k, rerank_top_k, final_context_chunks) are explicit.

Degradation (CLAUDE.md): no chunks -> grounded refusal (no LLM call); reranker down -> fall back to
dense order; embed/LLM down -> ExternalServiceUnavailable (the tool layer turns this into a
ToolError).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.prompts import load_prompt
from app.domain.exceptions import ExternalServiceUnavailable
from app.domain.rag import RetrievedChunk
from app.domain.tools import AnswerWithRagInput, AnswerWithRagOutput, Citation
from app.infra.errors import LLMError, ModelServerError
from app.infra.llm import LLMClient
from app.infra.model_server_client import RerankHit
from app.infra.redaction import redact_value
from app.infra.snapshot import NullSnapshotSink, SnapshotSink
from app.repositories.chunk_repository import StoredChunk

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = "rag_answer_v1.md"
_NO_CONTEXT_ANSWER = (
    "I couldn't find relevant information in the indexed docs or resolved issues to answer that. "
    "Try rephrasing or adding more detail."
)


@dataclass(frozen=True)
class RagParams:
    dense_top_k: int = 20
    rerank_top_k: int = 10
    final_context_chunks: int = 5
    use_rerank: bool = True
    max_answer_tokens: int = 512


class ChunkReaderPort(Protocol):
    async def search_cosine(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        level: str = ...,
        source_type: str | None = ...,
    ) -> list[RetrievedChunk]: ...

    async def fetch_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]: ...


class ModelServerPort(Protocol):
    async def embed(self, texts: Sequence[str], *, is_query: bool = ...) -> list[list[float]]: ...

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = ...
    ) -> list[RerankHit]: ...


class RagService:
    def __init__(
        self,
        *,
        chunks: ChunkReaderPort,
        model_server: ModelServerPort,
        llm: LLMClient,
        params: RagParams | None = None,
        snapshot_sink: SnapshotSink | None = None,
    ) -> None:
        self._chunks = chunks
        self._model_server = model_server
        self._llm = llm
        self._params = params or RagParams()
        self._snapshot_sink = snapshot_sink or NullSnapshotSink()

    async def answer(
        self, payload: AnswerWithRagInput, *, request_id: str | None = None
    ) -> AnswerWithRagOutput:
        query_vector = await self._embed_query(payload.question)

        retrieved = await self._chunks.search_cosine(
            query_vector,
            limit=self._params.dense_top_k,
            level="child",
            source_type=payload.source_type,
        )
        if not retrieved:
            return AnswerWithRagOutput(answer=_NO_CONTEXT_ANSWER, citations=[])

        child_meta = await self._chunks.fetch_by_ids([c.chunk_id for c in retrieved])
        ordered = await self._maybe_rerank(payload.question, retrieved, child_meta)
        final = ordered[: self._params.final_context_chunks]

        context, citations = await self._build_context_and_citations(final, child_meta)
        await self._write_snapshot(payload.question, final, request_id)

        if not context.strip():
            return AnswerWithRagOutput(answer=_NO_CONTEXT_ANSWER, citations=citations)

        answer = await self._generate(payload.question, context)
        return AnswerWithRagOutput(answer=answer, citations=citations)

    async def _embed_query(self, question: str) -> list[float]:
        try:
            vectors = await self._model_server.embed([question], is_query=True)
        except ModelServerError as exc:
            raise ExternalServiceUnavailable("The embedding service is unavailable.") from exc
        if not vectors or not vectors[0]:
            raise ExternalServiceUnavailable("The embedding service returned no vector.")
        return vectors[0]

    async def _maybe_rerank(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        child_meta: dict[str, StoredChunk],
    ) -> list[RetrievedChunk]:
        if not self._params.use_rerank:
            return retrieved
        candidates = [c for c in retrieved if c.chunk_id in child_meta]
        if not candidates:
            return retrieved
        passages = [child_meta[c.chunk_id].text for c in candidates]
        try:
            hits = await self._model_server.rerank(
                question, passages, top_k=self._params.rerank_top_k
            )
        except ModelServerError:
            logger.warning("rerank_unavailable_falling_back_to_dense")
            return retrieved
        return [candidates[hit.index] for hit in hits if 0 <= hit.index < len(candidates)]

    async def _build_context_and_citations(
        self, final: list[RetrievedChunk], child_meta: dict[str, StoredChunk]
    ) -> tuple[str, list[Citation]]:
        context_ids: list[str] = []
        for chunk in final:
            parent_id = chunk.parent_id or chunk.chunk_id
            if parent_id not in context_ids:
                context_ids.append(parent_id)
        context_meta = await self._chunks.fetch_by_ids(context_ids)
        context = "\n\n---\n\n".join(
            context_meta[pid].text for pid in context_ids if pid in context_meta
        )

        citations: list[Citation] = []
        for chunk in final:
            meta = child_meta.get(chunk.chunk_id)
            if meta is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_type=meta.source_type,
                    title=meta.title,
                    url=meta.url,
                    score=chunk.score,
                )
            )
        return context, citations

    async def _generate(self, question: str, context: str) -> str:
        prompt = (
            load_prompt(_ANSWER_PROMPT)
            .replace("{question}", question)
            .replace("{context}", context)
        )
        try:
            return await self._llm.generate(prompt, max_tokens=self._params.max_answer_tokens)
        except LLMError as exc:
            raise ExternalServiceUnavailable("Answer generation is unavailable.") from exc

    async def _write_snapshot(
        self, question: str, final: list[RetrievedChunk], request_id: str | None
    ) -> None:
        payload = redact_value(
            {
                "request_id": request_id,
                "question": question,
                "retrieved_chunk_ids": [c.chunk_id for c in final],
                "scores": [c.score for c in final],
            }
        )
        await self._snapshot_sink.write_snapshot(f"rag/{request_id or 'adhoc'}.json", payload)
