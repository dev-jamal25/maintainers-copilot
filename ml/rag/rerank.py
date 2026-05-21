"""Cross-encoder reranking over hybrid candidates (A10, DECISIONS D1.9 / D3.10).

First retrieve broadly (hybrid), then rerank the candidates with a stronger pairwise model
(MS MARCO MiniLM cross-encoder) and keep the top ``rerank_top_k`` (default 10).

The ordering logic takes an injected ``score_fn`` so it is unit-tested offline; the real model
is loaded only by ``CrossEncoderReranker`` / ``get_reranker`` for the eval and runtime paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

from backend.app.domain.rag import RetrievedChunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 10

# (query, passages) -> one relevance score per passage.
CrossScoreFn = Callable[[str, Sequence[str]], list[float]]


class CrossEncoderReranker:
    """Lazily loads the MS MARCO MiniLM cross-encoder and scores (query, passage) pairs."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        return [float(value) for value in self._model.predict(pairs)]


@lru_cache(maxsize=2)
def get_reranker(model_name: str = RERANK_MODEL) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def rerank(
    query: str,
    candidates: Sequence[RetrievedChunk],
    text_by_id: Mapping[str, str],
    score_fn: CrossScoreFn,
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[RetrievedChunk]:
    """Re-score candidates with ``score_fn`` and return the top_k, re-ranked from 0.

    Ties are broken by the candidate's original (pre-rerank) rank for determinism.
    """
    if not candidates:
        return []
    passages = [text_by_id.get(candidate.chunk_id, "") for candidate in candidates]
    scores = score_fn(query, passages)
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
    return [
        RetrievedChunk(
            chunk_id=candidates[i].chunk_id,
            parent_id=candidates[i].parent_id,
            score=float(scores[i]),
            rank=new_rank,
        )
        for new_rank, i in enumerate(order[:top_k])
    ]
