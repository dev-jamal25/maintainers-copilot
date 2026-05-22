"""Embedding service: BAAI/bge-small-en-v1.5, 384-dim, L2-normalized (frozen DECISIONS choice).

The encoder is loaded lazily and cached (no import-time globals, no downloads in CI — tests
monkeypatch ``build_encoder``). bge is asymmetric: queries get an instruction prefix, passages
do not. Vectors are L2-normalized so cosine similarity equals a dot product downstream (pgvector).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final, Protocol, cast

from app.schemas.embed import EmbedResponse

EMBED_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"
EMBED_DIM: Final[int] = 384
QUERY_PREFIX: Final[str] = "Represent this sentence for searching relevant passages: "
MAX_SEQUENCE_TOKENS: Final[int] = 512


class EncoderCallable(Protocol):
    def __call__(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        """Return one L2-normalized embedding per input text."""


class TransformersEncoder:
    """Real bge encoder: transformers + mean pooling. Constructed lazily (downloads weights)."""

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        self._model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        self._model.eval()

    def __call__(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        import torch

        prepared = [QUERY_PREFIX + text for text in texts] if is_query else list(texts)
        encoded = self._tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        normalized = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return cast("list[list[float]]", normalized.tolist())


def build_encoder() -> EncoderCallable:
    return TransformersEncoder()


@lru_cache(maxsize=1)
def get_encoder() -> EncoderCallable:
    return build_encoder()


def embed_texts(texts: Sequence[str], *, is_query: bool = False) -> EmbedResponse:
    vectors = get_encoder()(texts, is_query=is_query)
    return EmbedResponse(embeddings=vectors, model=EMBED_MODEL_NAME, dim=EMBED_DIM)
