from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import (  # noqa: E402
    ChunkLevel,
    ChunkSourceType,
    CorpusRecord,
)

from rag.chunking import (  # noqa: E402
    ChunkingParams,
    chunk_corpus_record,
    naive_chunk_corpus_record,
    pack_segments,
    split_blocks,
    split_doc_sections,
)

# Small word-count counter keeps chunk boundaries deterministic and offline.
PARAMS = ChunkingParams(
    max_child_tokens=20, min_child_tokens=10, overlap_tokens=5, naive_chunk_tokens=10
)


def _wc(text: str) -> int:
    return len(text.split())


def _doc_record(text: str) -> CorpusRecord:
    return CorpusRecord(
        source_type=ChunkSourceType.DOCS,
        source_id="dags",
        text=text,
        title="DAGs",
        url="https://example/dags",
        airflow_area="core_concepts",
        tags=["core_concepts"],
        version="2.10.3",
    )


def _issue_record(text: str) -> CorpusRecord:
    return CorpusRecord(
        source_type=ChunkSourceType.ISSUE,
        source_id="issue:111:comment:0",
        text=text,
        title="Scheduler stuck",
        tags=["bug"],
        github_issue_id=111,
        github_comment_id=0,
    )


def test_split_doc_sections_tracks_heading_breadcrumb() -> None:
    text = "# DAGs\n\nIntro.\n\n## Declaring\n\nbody1\n\n### Sub\n\nbody2\n"
    sections = split_doc_sections(text)
    paths = [path for path, _ in sections]
    assert paths == [["DAGs"], ["DAGs", "Declaring"], ["DAGs", "Declaring", "Sub"]]
    assert "Intro." in sections[0][1]


def test_split_blocks_keeps_fenced_code_whole() -> None:
    text = "para one\n\n```\ncode a\ncode b\n```\n\npara two\n"
    blocks = split_blocks(text)
    assert blocks[0] == "para one"
    assert "```\ncode a\ncode b\n```" in blocks
    assert "para two" in blocks


def test_pack_segments_respects_max_and_carries_overlap() -> None:
    segments = ["aa bb cc dd ee", "ff gg hh ii jj", "kk ll mm nn oo"]
    chunks = pack_segments(segments, max_tokens=10, overlap_tokens=5, count=_wc, joiner=" ")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert _wc(chunk) <= 10
    # The middle segment is carried as overlap into the next chunk.
    assert "ff gg hh ii jj" in chunks[0]
    assert "ff gg hh ii jj" in chunks[1]


def test_chunk_doc_builds_parents_children_with_stable_ids() -> None:
    text = "# DAGs\n\n" + " ".join(["word"] * 30) + "\n\n## Sub\n\n" + " ".join(["x"] * 30) + "\n"
    chunks = chunk_corpus_record(_doc_record(text), PARAMS, _wc)
    parents = [c for c in chunks if c.level == ChunkLevel.PARENT]
    children = [c for c in chunks if c.level == ChunkLevel.CHILD]
    assert [p.chunk_id for p in parents] == ["doc:dags:parent:0", "doc:dags:parent:1"]
    assert any(c.chunk_id == "doc:dags:parent:0:child:0" for c in children)
    for child in children:
        assert child.parent_id in {"doc:dags:parent:0", "doc:dags:parent:1"}
        assert _wc(child.text) <= PARAMS.max_child_tokens
    sub = [c for c in children if c.parent_id == "doc:dags:parent:1"]
    assert sub and sub[0].metadata.section_path == ["DAGs", "Sub"]


def test_chunk_issue_is_single_parent_with_comment_scoped_ids() -> None:
    chunks = chunk_corpus_record(_issue_record(" ".join(["w"] * 30)), PARAMS, _wc)
    parents = [c for c in chunks if c.level == ChunkLevel.PARENT]
    children = [c for c in chunks if c.level == ChunkLevel.CHILD]
    assert len(parents) == 1
    assert parents[0].chunk_id == "issue:111:comment:0:parent:0"
    assert any(c.chunk_id == "issue:111:comment:0:child:0" for c in children)
    for child in children:
        assert child.parent_id == "issue:111:comment:0:parent:0"
        assert child.metadata.github_issue_id == 111
        assert _wc(child.text) <= PARAMS.max_child_tokens


def test_naive_chunks_are_flat_children_with_naive_ids() -> None:
    chunks = naive_chunk_corpus_record(_doc_record(" ".join(["t"] * 25)), PARAMS, _wc)
    assert chunks, "naive chunker must emit chunks"
    assert all(c.level == ChunkLevel.CHILD for c in chunks)
    assert all(c.parent_id is None for c in chunks)
    assert chunks[0].chunk_id == "doc:dags:naive:0"
    for chunk in chunks:
        assert _wc(chunk.text) <= PARAMS.naive_chunk_tokens


def test_chunking_is_deterministic_across_runs() -> None:
    record = _issue_record(" ".join(["alpha", "beta"] * 25))
    first = chunk_corpus_record(record, PARAMS, _wc)
    second = chunk_corpus_record(record, PARAMS, _wc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
