"""Cross-encoder reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (frozen DECISIONS choice).

Scores each (query, passage) pair and returns passages in descending relevance. Loaded lazily +
cached (tests monkeypatch ``build_reranker`` so CI runs without downloads).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final, Protocol, cast

from app.schemas.rerank import RerankResponse, RerankResult

RERANK_MODEL_NAME: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQUENCE_TOKENS: Final[int] = 512


class RerankerCallable(Protocol):
    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage (higher = more relevant)."""


class TransformersReranker:
    """Real cross-encoder using a sequence-classification head. Constructed lazily."""

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
        self._model.eval()

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch

        pairs = [[query, passage] for passage in passages]
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        scores = logits.squeeze(-1)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        return cast("list[float]", scores.tolist())


def build_reranker() -> RerankerCallable:
    return TransformersReranker()


@lru_cache(maxsize=1)
def get_reranker() -> RerankerCallable:
    return build_reranker()


def rerank_passages(
    query: str, passages: Sequence[str], *, top_k: int | None = None
) -> RerankResponse:
    scores = get_reranker()(query, passages)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    results = [RerankResult(index=index, score=float(score)) for index, score in ranked]
    return RerankResponse(results=results, model=RERANK_MODEL_NAME)
