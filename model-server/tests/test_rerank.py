from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import rerank


class FakeReranker:
    """Token-overlap scorer: more shared words with the query => higher score (no download)."""

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        return [float(len(query_tokens & set(p.lower().split()))) for p in passages]


@pytest.fixture(autouse=True)
def clear_reranker_cache() -> None:
    rerank.get_reranker.cache_clear()


def test_rerank_orders_by_relevance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank, "build_reranker", FakeReranker)
    response = rerank.rerank_passages(
        "scheduler restart loop",
        ["unrelated docs about ui", "the scheduler enters a restart loop on oom"],
    )
    assert response.model == rerank.RERANK_MODEL_NAME
    assert response.results[0].index == 1  # most relevant first
    assert response.results[0].score >= response.results[1].score


def test_rerank_respects_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank, "build_reranker", FakeReranker)
    response = rerank.rerank_passages("dag", ["dag a", "dag b", "nope"], top_k=2)
    assert len(response.results) == 2


def test_rerank_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank, "build_reranker", FakeReranker)
    client = TestClient(app)
    response = client.post(
        "/rerank",
        json={"query": "scheduler", "passages": ["scheduler hangs", "ui theme"], "top_k": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["index"] == 0


def test_rerank_endpoint_rejects_empty() -> None:
    client = TestClient(app)
    assert client.post("/rerank", json={"query": "x", "passages": []}).status_code == 422
    assert client.post("/rerank", json={"query": "  ", "passages": ["a"]}).status_code == 422
