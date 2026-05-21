"""Dense + sparse + hybrid retrieval over child chunks (A09, DECISIONS D3.8 / D3.10).

- Dense: cosine similarity (dot product of L2-normalized embeddings).
- Sparse: rank-bm25 (true BM25Okapi) over whitespace/alnum-tokenized passages.
- Hybrid: per-query min-max normalize each modality over the candidate union, then
  ``dense_weight * dense + sparse_weight * sparse``; tuned by the golden-set weight sweep.

Defaults (D3.10): dense top_k = 20, sparse top_k = 20, hybrid merged top_k = 30,
final context chunks = 5. Weight sweep (D3.8): dense/sparse 0.25/0.75, 0.50/0.50, 0.75/0.25.

Retrieval runs on *children*; results expose the parent IDs to expand for generation context
(small-to-big). The index takes precomputed embeddings so it is testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from backend.app.domain.rag import Chunk, ChunkLevel, RetrievalResult, RetrievedChunk
from numpy.typing import NDArray

from rag.embeddings import EmbeddingMatrix

Vector = NDArray[np.float32]
Scores = NDArray[np.float64]


@dataclass(frozen=True)
class RetrievalParams:
    """Retrieval knobs (D3.10 defaults)."""

    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 30
    final_context_chunks: int = 5


DEFAULT_RETRIEVAL_PARAMS = RetrievalParams()

# (dense_weight, sparse_weight) sweep points (D3.8).
WEIGHT_SWEEP: tuple[tuple[float, float], ...] = ((0.25, 0.75), (0.50, 0.50), (0.75, 0.25))

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return _TOKEN_RE.findall(text.lower())


def _top_indices(scores: Scores, k: int) -> list[int]:
    """Indices of the top-k scores, descending, with stable index tie-breaking."""
    if len(scores) == 0:
        return []
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order]


def _minmax_over(scores: Scores, indices: list[int]) -> dict[int, float]:
    """Min-max normalize ``scores`` for the given candidate indices into [0, 1]."""
    if not indices:
        return {}
    values = scores[indices]
    low = float(values.min())
    high = float(values.max())
    span = high - low
    if span <= 0.0:
        return {index: 0.0 for index in indices}
    return {index: (float(scores[index]) - low) / span for index in indices}


class RetrievalIndex:
    """In-memory dense + BM25 index over child chunks (embeddings precomputed)."""

    def __init__(
        self,
        chunk_ids: list[str],
        embeddings: EmbeddingMatrix,
        texts: list[str],
        parent_ids: list[str | None],
    ) -> None:
        if not (len(chunk_ids) == embeddings.shape[0] == len(texts) == len(parent_ids)):
            raise ValueError("chunk_ids, embeddings, texts, parent_ids must align")
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        self.embeddings = embeddings
        self.parent_ids = parent_ids
        self._bm25 = BM25Okapi([tokenize(text) for text in texts])

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def dense_scores(self, query_vector: Vector) -> Scores:
        return np.asarray(self.embeddings @ query_vector, dtype=np.float64)

    def sparse_scores(self, query_text: str) -> Scores:
        return np.asarray(self._bm25.get_scores(tokenize(query_text)), dtype=np.float64)

    def _build_result(
        self,
        query: str,
        ranked: list[int],
        score_map: dict[int, float],
        params: RetrievalParams,
    ) -> RetrievalResult:
        retrieved = [
            RetrievedChunk(
                chunk_id=self.chunk_ids[index],
                parent_id=self.parent_ids[index],
                score=score_map[index],
                rank=rank,
            )
            for rank, index in enumerate(ranked)
        ]
        context_parent_ids: list[str] = []
        for chunk in retrieved[: params.final_context_chunks]:
            parent = chunk.parent_id or chunk.chunk_id
            if parent not in context_parent_ids:
                context_parent_ids.append(parent)
        return RetrievalResult(
            query=query, retrieved=retrieved, context_parent_ids=context_parent_ids
        )

    def search_dense(
        self,
        query_vector: Vector,
        query_text: str = "",
        params: RetrievalParams = DEFAULT_RETRIEVAL_PARAMS,
    ) -> RetrievalResult:
        scores = self.dense_scores(query_vector)
        ranked = _top_indices(scores, params.hybrid_top_k)
        score_map = {index: float(scores[index]) for index in ranked}
        return self._build_result(query_text, ranked, score_map, params)

    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        *,
        dense_weight: float,
        sparse_weight: float,
        params: RetrievalParams = DEFAULT_RETRIEVAL_PARAMS,
    ) -> RetrievalResult:
        dense = self.dense_scores(query_vector)
        sparse = self.sparse_scores(query_text)
        candidates = sorted(
            set(_top_indices(dense, params.dense_top_k))
            | set(_top_indices(sparse, params.sparse_top_k))
        )
        norm_dense = _minmax_over(dense, candidates)
        norm_sparse = _minmax_over(sparse, candidates)
        combined = {
            index: dense_weight * norm_dense[index] + sparse_weight * norm_sparse[index]
            for index in candidates
        }
        ranked = sorted(candidates, key=lambda index: (-combined[index], index))
        ranked = ranked[: params.hybrid_top_k]
        return self._build_result(query_text, ranked, combined, params)


def build_retrieval_index(
    child_chunks: list[Chunk],
    embeddings: EmbeddingMatrix,
) -> RetrievalIndex:
    """Build an index from child chunks + their aligned embedding matrix."""
    children = [chunk for chunk in child_chunks if chunk.level == ChunkLevel.CHILD]
    if len(children) != embeddings.shape[0]:
        raise ValueError("embeddings must align 1:1 with child chunks")
    return RetrievalIndex(
        chunk_ids=[chunk.chunk_id for chunk in children],
        embeddings=embeddings,
        texts=[chunk.text for chunk in children],
        parent_ids=[chunk.parent_id for chunk in children],
    )
