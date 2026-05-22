"""Model-server client: success parsing, bounded retry on 5xx/transport errors, 4xx no-retry."""

from __future__ import annotations

import httpx
import pytest

from app.core.config import ModelServerSettings
from app.infra.errors import ModelServerResponseError, ModelServerUnavailableError
from app.infra.model_server_client import ModelServerClient


def _client(handler: httpx.MockTransport, *, max_retries: int = 2) -> ModelServerClient:
    settings = ModelServerSettings(max_retries=max_retries)
    transport_client = httpx.AsyncClient(transport=handler, base_url="http://model-server:8001")
    return ModelServerClient(settings, client=transport_client)


@pytest.mark.asyncio
async def test_embed_parses_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]], "model": "bge", "dim": 2})

    client = _client(httpx.MockTransport(handler))
    vectors = await client.embed(["hello"], is_query=True)
    assert vectors == [[0.1, 0.2]]
    await client.aclose()


@pytest.mark.asyncio
async def test_rerank_parses_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"index": 1, "score": 9.0}, {"index": 0, "score": 1.0}]}
        )

    client = _client(httpx.MockTransport(handler))
    hits = await client.rerank("q", ["a", "b"], top_k=2)
    assert [(h.index, h.score) for h in hits] == [(1, 9.0), (0, 1.0)]
    await client.aclose()


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"embeddings": [[1.0]], "model": "bge", "dim": 1})

    client = _client(httpx.MockTransport(handler), max_retries=2)
    vectors = await client.embed(["x"])
    assert vectors == [[1.0]]
    assert calls["n"] == 3  # two failures + one success
    await client.aclose()


@pytest.mark.asyncio
async def test_persistent_5xx_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client(httpx.MockTransport(handler), max_retries=1)
    with pytest.raises(ModelServerUnavailableError):
        await client.embed(["x"])
    await client.aclose()


@pytest.mark.asyncio
async def test_4xx_raises_response_error_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad"})

    client = _client(httpx.MockTransport(handler), max_retries=3)
    with pytest.raises(ModelServerResponseError):
        await client.embed(["x"])
    assert calls["n"] == 1  # 4xx is not retried
    await client.aclose()


@pytest.mark.asyncio
async def test_classify_parses_label_and_confidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/classify"
        return httpx.Response(200, json={"label": "bug", "scores": {"bug": 0.7, "docs": 0.3}})

    client = _client(httpx.MockTransport(handler))
    label, confidence = await client.classify("scheduler crashes", "traceback")
    assert label == "bug"
    assert confidence == 0.7
    await client.aclose()


@pytest.mark.asyncio
async def test_extract_entities_maps_text_and_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"entities": [{"text": "Airflow", "type": "ORG", "start": 0, "end": 7}]},
        )

    client = _client(httpx.MockTransport(handler))
    entities = await client.extract_entities("Airflow scheduler")
    assert entities == [{"text": "Airflow", "type": "ORG"}]
    await client.aclose()


@pytest.mark.asyncio
async def test_transport_error_retried_then_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(httpx.MockTransport(handler), max_retries=1)
    with pytest.raises(ModelServerUnavailableError):
        await client.rerank("q", ["a"])
    await client.aclose()
