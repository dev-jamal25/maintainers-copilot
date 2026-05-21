from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import (  # noqa: E402
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkSourceType,
)

from rag.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_PARAMS,
    WEIGHT_SWEEP,
    RetrievalIndex,
    RetrievalParams,
    build_retrieval_index,
    tokenize,
)

PARAMS = RetrievalParams(dense_top_k=4, sparse_top_k=4, hybrid_top_k=4, final_context_chunks=2)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def _index() -> RetrievalIndex:
    embeddings = np.stack(
        [_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1]), _unit([0.9, 0.1, 0.0])]
    ).astype(np.float32)
    texts = [
        "scheduler stuck dagrun",
        "celery worker queue",
        "webserver ui login",
        "scheduler restart fix",
    ]
    return RetrievalIndex(
        chunk_ids=["c0", "c1", "c2", "c3"],
        embeddings=embeddings,
        texts=texts,
        parent_ids=["P0", "P1", "P2", "P0"],
    )


def test_default_params_match_decisions_d310() -> None:
    assert DEFAULT_RETRIEVAL_PARAMS.dense_top_k == 20
    assert DEFAULT_RETRIEVAL_PARAMS.sparse_top_k == 20
    assert DEFAULT_RETRIEVAL_PARAMS.hybrid_top_k == 30
    assert DEFAULT_RETRIEVAL_PARAMS.final_context_chunks == 5


def test_weight_sweep_matches_decisions_d38() -> None:
    assert WEIGHT_SWEEP == ((0.25, 0.75), (0.50, 0.50), (0.75, 0.25))


def test_tokenize_is_lowercase_alphanumeric() -> None:
    assert tokenize("Scheduler STUCK, dag-run!") == ["scheduler", "stuck", "dag", "run"]


def test_dense_search_ranks_by_cosine() -> None:
    result = _index().search_dense(_unit([1, 0, 0]), "scheduler", PARAMS)
    assert result.retrieved[0].chunk_id == "c0"
    assert result.retrieved[1].chunk_id == "c3"  # nearest neighbour of [1,0,0]


def test_hybrid_sparse_only_ranks_by_bm25() -> None:
    result = _index().search_hybrid(
        _unit([0, 0, 1]),
        "scheduler restart fix",
        dense_weight=0.0,
        sparse_weight=1.0,
        params=PARAMS,
    )
    assert result.retrieved[0].chunk_id == "c3"  # most BM25 term overlap


def test_hybrid_respects_top_k_and_expands_parents() -> None:
    result = _index().search_hybrid(
        _unit([1, 0, 0]),
        "scheduler",
        dense_weight=0.5,
        sparse_weight=0.5,
        params=PARAMS,
    )
    assert len(result.retrieved) <= PARAMS.hybrid_top_k
    assert all(rc.rank == i for i, rc in enumerate(result.retrieved))
    # context expands children -> parents, deduped, capped at final_context_chunks.
    assert len(result.context_parent_ids) <= PARAMS.final_context_chunks
    assert result.context_parent_ids == list(dict.fromkeys(result.context_parent_ids))


def test_build_retrieval_index_aligns_children_and_embeddings() -> None:
    children = [
        Chunk(
            level=ChunkLevel.CHILD,
            text="scheduler stuck",
            metadata=ChunkMetadata(
                chunk_id="doc:x:parent:0:child:0",
                parent_id="doc:x:parent:0",
                source_type=ChunkSourceType.DOCS,
                source_id="x",
            ),
        )
    ]
    embeddings = np.stack([_unit([1, 0, 0])]).astype(np.float32)
    index = build_retrieval_index(children, embeddings)
    assert len(index) == 1
    assert index.parent_ids == ["doc:x:parent:0"]

    with pytest.raises(ValueError, match="align"):
        build_retrieval_index(children, np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])]))
