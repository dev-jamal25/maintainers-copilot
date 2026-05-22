"""Async HTTP adapter for the model-server boundary (CLAUDE.md).

`api` never imports model-server modules — it talks HTTP via this adapter using
``httpx.AsyncClient`` with a timeout and bounded retry. Transient failures (transport errors, 5xx)
are retried with a short async backoff; on exhaustion we raise ``ModelServerUnavailableError``. 4xx
and unparseable bodies raise ``ModelServerResponseError`` (no retry). Services convert these infra
errors into domain exceptions / structured tool errors so the chatbot degrades gracefully.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.config import ModelServerSettings
from app.infra.errors import ModelServerResponseError, ModelServerUnavailableError

_BACKOFF_BASE_SECONDS = 0.2


class RerankHit:
    """A single rerank result: original passage index + relevance score."""

    __slots__ = ("index", "score")

    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.score = score


class ModelServerClient:
    def __init__(
        self,
        settings: ModelServerSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._max_retries = max(0, settings.max_retries)
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url, timeout=settings.timeout_seconds
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TransportError as exc:  # connect/read/timeout — transient
                last_exc = exc
            else:
                if response.status_code < 400:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise ModelServerResponseError(
                            f"model-server {path} returned non-JSON body"
                        ) from exc
                    if not isinstance(body, dict):
                        raise ModelServerResponseError(
                            f"model-server {path} returned a non-object body"
                        )
                    return body
                if response.status_code < 500:
                    raise ModelServerResponseError(
                        f"model-server {path} returned {response.status_code}"
                    )
                last_exc = ModelServerUnavailableError(
                    f"model-server {path} returned {response.status_code}"
                )
            if attempt < self._max_retries:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        raise ModelServerUnavailableError(
            f"model-server {path} unavailable after {self._max_retries + 1} attempts"
        ) from last_exc

    async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        body = await self._post_json("/embed", {"texts": list(texts), "is_query": is_query})
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise ModelServerResponseError("model-server /embed missing 'embeddings'")
        return [[float(value) for value in vector] for vector in embeddings]

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankHit]:
        payload: dict[str, Any] = {"query": query, "passages": list(passages)}
        if top_k is not None:
            payload["top_k"] = top_k
        body = await self._post_json("/rerank", payload)
        results = body.get("results")
        if not isinstance(results, list):
            raise ModelServerResponseError("model-server /rerank missing 'results'")
        return [RerankHit(int(item["index"]), float(item["score"])) for item in results]

    async def summarize(self, text: str) -> str:
        body = await self._post_json("/summarize", {"text": text})
        summary = body.get("summary")
        if not isinstance(summary, str):
            raise ModelServerResponseError("model-server /summarize missing 'summary'")
        return summary
