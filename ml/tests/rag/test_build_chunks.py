from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import (  # noqa: E402
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkSourceType,
    CorpusRecord,
)

from rag.build_chunks import (  # noqa: E402
    assert_parents_exist,
    assert_unique_ids,
    build_advanced_chunks,
    build_manifest,
    build_naive_chunks,
)
from rag.chunking import ChunkingParams  # noqa: E402

PARAMS = ChunkingParams(
    max_child_tokens=20, min_child_tokens=10, overlap_tokens=5, naive_chunk_tokens=10
)


def _wc(text: str) -> int:
    return len(text.split())


def _records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            source_type=ChunkSourceType.DOCS,
            source_id="dags",
            text="# DAGs\n\n" + " ".join(["word"] * 30) + "\n\n## Sub\n\n" + " ".join(["x"] * 30),
            title="DAGs",
            airflow_area="core_concepts",
            tags=["core_concepts"],
            version="2.10.3",
        ),
        CorpusRecord(
            source_type=ChunkSourceType.ISSUE,
            source_id="issue:111:comment:0",
            text=" ".join(["w"] * 40),
            title="Scheduler stuck",
            tags=["bug"],
            github_issue_id=111,
            github_comment_id=0,
        ),
    ]


def test_advanced_chunks_have_unique_ids_and_resolvable_parents() -> None:
    chunks = build_advanced_chunks(_records(), PARAMS, _wc)
    assert any(c.level == ChunkLevel.PARENT for c in chunks)
    assert any(c.level == ChunkLevel.CHILD for c in chunks)
    assert_unique_ids(chunks, label="advanced")  # no raise
    assert_parents_exist(chunks)  # no raise


def test_naive_chunks_are_flat_and_unique() -> None:
    chunks = build_naive_chunks(_records(), PARAMS, _wc)
    assert chunks
    assert all(c.level == ChunkLevel.CHILD for c in chunks)
    assert_unique_ids(chunks, label="naive")  # no raise


def test_assert_unique_ids_detects_duplicates() -> None:
    meta = ChunkMetadata(chunk_id="doc:x:parent:0", source_type=ChunkSourceType.DOCS, source_id="x")
    dup = [
        Chunk(level=ChunkLevel.PARENT, text="a", metadata=meta),
        Chunk(level=ChunkLevel.PARENT, text="b", metadata=meta),
    ]
    with pytest.raises(ValueError, match="Duplicate advanced chunk IDs"):
        assert_unique_ids(dup, label="advanced")


def test_assert_parents_exist_detects_orphans() -> None:
    orphan_meta = ChunkMetadata(
        chunk_id="doc:x:parent:0:child:0",
        parent_id="doc:x:parent:9",  # no such parent in the set
        source_type=ChunkSourceType.DOCS,
        source_id="x",
    )
    orphan = [Chunk(level=ChunkLevel.CHILD, text="c", metadata=orphan_meta)]
    with pytest.raises(ValueError, match="missing parents"):
        assert_parents_exist(orphan)


def test_build_manifest_counts_levels() -> None:
    records = _records()
    advanced = build_advanced_chunks(records, PARAMS, _wc)
    naive = build_naive_chunks(records, PARAMS, _wc)
    doc_records = [r for r in records if r.source_type == ChunkSourceType.DOCS]
    issue_records = [r for r in records if r.source_type == ChunkSourceType.ISSUE]
    manifest = build_manifest(
        doc_records,
        issue_records,
        advanced,
        naive,
        PARAMS,
        tokenizer="wordcount-fallback",
        generated_at="2026-05-21T00:00:00+00:00",
    )
    assert manifest["doc_records"] == 1
    assert manifest["issue_records"] == 1
    assert manifest["advanced"]["total"] == len(advanced)
    assert manifest["advanced"]["parents"] + manifest["advanced"]["children"] == len(advanced)
    assert manifest["naive"]["total"] == len(naive)
    assert manifest["chunking_params"]["max_child_tokens"] == 20
