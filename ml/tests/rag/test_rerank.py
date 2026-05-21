from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import RetrievedChunk  # noqa: E402

from rag.rerank import DEFAULT_RERANK_TOP_K, rerank  # noqa: E402


def _candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id="a", parent_id="P0", score=0.9, rank=0),
        RetrievedChunk(chunk_id="b", parent_id="P1", score=0.8, rank=1),
        RetrievedChunk(chunk_id="c", parent_id="P2", score=0.7, rank=2),
    ]


_TEXT = {"a": "alpha", "b": "beta", "c": "gamma"}


def test_default_rerank_top_k_matches_decisions() -> None:
    assert DEFAULT_RERANK_TOP_K == 10


def test_rerank_reorders_by_cross_encoder_score_and_caps_top_k() -> None:
    def score_fn(query: str, passages: Sequence[str]) -> list[float]:
        scores = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}
        return [scores[p] for p in passages]

    out = rerank("q", _candidates(), _TEXT, score_fn, top_k=2)
    assert [c.chunk_id for c in out] == ["b", "c"]
    assert [c.rank for c in out] == [0, 1]
    assert out[0].score == 0.9


def test_rerank_breaks_ties_by_original_rank() -> None:
    def score_fn(query: str, passages: Sequence[str]) -> list[float]:
        return [0.5 for _ in passages]  # all tied

    out = rerank("q", _candidates(), _TEXT, score_fn, top_k=3)
    assert [c.chunk_id for c in out] == ["a", "b", "c"]  # stable by original rank


def test_rerank_empty_candidates() -> None:
    def score_fn(query: str, passages: Sequence[str]) -> list[float]:
        return []

    assert rerank("q", [], {}, score_fn) == []
