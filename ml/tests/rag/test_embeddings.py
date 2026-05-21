from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.embeddings import (  # noqa: E402
    BGE_QUERY_INSTRUCTION,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODELS,
    SentenceTransformerEmbedder,
    apply_query_instruction,
    get_embedder,
)


def test_registry_has_both_compared_models() -> None:
    assert set(EMBEDDING_MODELS) == {"bge-small", "minilm"}
    assert EMBEDDING_MODELS["bge-small"] == "BAAI/bge-small-en-v1.5"
    assert EMBEDDING_MODELS["minilm"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert EMBEDDING_DIM == 384
    assert DEFAULT_EMBEDDING_MODEL == "bge-small"


def test_bge_query_instruction_is_prepended_for_queries() -> None:
    out = apply_query_instruction(["how to fix the scheduler"], "bge-small")
    assert out == [f"{BGE_QUERY_INSTRUCTION} how to fix the scheduler"]


def test_minilm_queries_are_not_prefixed() -> None:
    out = apply_query_instruction(["how to fix the scheduler"], "minilm")
    assert out == ["how to fix the scheduler"]


def test_unknown_model_key_raises_before_loading() -> None:
    # Guard must fire before importing/loading sentence-transformers (no network/download).
    with pytest.raises(ValueError, match="Unknown embedding model"):
        SentenceTransformerEmbedder("does-not-exist")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_embedder("does-not-exist")
