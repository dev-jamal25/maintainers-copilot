"""Embedding models for RAG (A08, DECISIONS D3.3).

Wraps the two compared local embedding models behind one interface:
- ``bge-small`` -> ``BAAI/bge-small-en-v1.5`` (planned production model).
- ``minilm``    -> ``sentence-transformers/all-MiniLM-L6-v2`` (free/fast baseline).

Both output 384-dim vectors, so the pgvector column dimension is stable across the comparison.
Embeddings are L2-normalized so cosine similarity reduces to a dot product. bge-small expects a
query instruction prefix for retrieval (passages get none); MiniLM uses no prefix.

The final embedding choice must be backed by Hit@5 and MRR@10 on the golden set (D3.3); this
module is the swappable component the eval harness compares.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

EmbeddingMatrix = NDArray[np.float32]

EMBEDDING_MODELS: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_EMBEDDING_MODEL = "bge-small"
EMBEDDING_DIM = 384
# bge retrieval queries are prefixed with this instruction; passages are embedded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


def apply_query_instruction(texts: Sequence[str], model_key: str) -> list[str]:
    """Prepend the bge query instruction for query embedding; pass through for other models."""
    if model_key == "bge-small":
        return [f"{BGE_QUERY_INSTRUCTION} {text}" for text in texts]
    return list(texts)


class SentenceTransformerEmbedder:
    """Lazily loads a SentenceTransformer and returns normalized float32 embeddings."""

    def __init__(self, model_key: str) -> None:
        if model_key not in EMBEDDING_MODELS:
            choices = sorted(EMBEDDING_MODELS)
            raise ValueError(f"Unknown embedding model '{model_key}'; choose from {choices}")
        from sentence_transformers import SentenceTransformer

        self.model_key = model_key
        self._model = SentenceTransformer(EMBEDDING_MODELS[model_key])

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> EmbeddingMatrix:
        prepared = apply_query_instruction(list(texts), self.model_key) if is_query else list(texts)
        vectors = self._model.encode(prepared, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=4)
def get_embedder(model_key: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformerEmbedder:
    """Return a cached embedder for ``model_key`` (validates the key before any model load)."""
    return SentenceTransformerEmbedder(model_key)
