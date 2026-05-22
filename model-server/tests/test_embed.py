from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import embed


class FakeEncoder:
    """Deterministic 384-dim unit-vector encoder (no model download)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        self.calls += 1
        return [self._vector(("q:" if is_query else "p:") + text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(embed.EMBED_DIM)]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]


@pytest.fixture(autouse=True)
def clear_encoder_cache() -> None:
    embed.get_encoder.cache_clear()


def test_embed_returns_normalized_384_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "build_encoder", FakeEncoder)
    response = embed.embed_texts(["scheduler hangs", "dag import error"])
    assert response.model == embed.EMBED_MODEL_NAME
    assert response.dim == 384
    assert len(response.embeddings) == 2
    assert all(len(vector) == 384 for vector in response.embeddings)
    norm = sum(v * v for v in response.embeddings[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_query_prefix_changes_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "build_encoder", FakeEncoder)
    as_passage = embed.embed_texts(["airflow"], is_query=False).embeddings[0]
    as_query = embed.embed_texts(["airflow"], is_query=True).embeddings[0]
    assert as_passage != as_query


def test_encoder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeEncoder] = []

    def factory() -> FakeEncoder:
        encoder = FakeEncoder()
        created.append(encoder)
        return encoder

    monkeypatch.setattr(embed, "build_encoder", factory)
    embed.embed_texts(["a"])
    embed.embed_texts(["b"])
    assert len(created) == 1
    assert created[0].calls == 2


def test_embed_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "build_encoder", FakeEncoder)
    client = TestClient(app)
    response = client.post("/embed", json={"texts": ["scheduler hangs"], "is_query": True})
    assert response.status_code == 200
    body = response.json()
    assert body["dim"] == 384
    assert len(body["embeddings"][0]) == 384


def test_embed_endpoint_rejects_blank_and_empty() -> None:
    client = TestClient(app)
    assert client.post("/embed", json={"texts": []}).status_code == 422
    assert client.post("/embed", json={"texts": ["   "]}).status_code == 422
