from __future__ import annotations

import sys
from collections import Counter
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
    doc_child_id,
    doc_parent_id,
    issue_child_id,
    issue_parent_id,
)

from rag.golden import (  # noqa: E402
    EXPECTED_DISTRIBUTION,
    GOLDEN_SIZE,
    build_draft_golden,
    validate_golden_examples,
)

_TEXT = "This is a substantive maintainer answer that explains the fix in enough detail. " * 2


def _issue_unit(github_id: int, comment_id: int) -> list[Chunk]:
    source_id = f"issue:{github_id}:comment:{comment_id}"
    parent_id = issue_parent_id(github_id, comment_id, 0)
    child_id = issue_child_id(github_id, comment_id, 0)
    common = {
        "source_type": ChunkSourceType.ISSUE,
        "source_id": source_id,
        "title": f"Issue {github_id} is broken",
        "github_issue_id": github_id,
        "github_comment_id": comment_id,
    }
    return [
        Chunk(
            level=ChunkLevel.PARENT,
            text=_TEXT,
            metadata=ChunkMetadata(chunk_id=parent_id, parent_id=None, tags=["bug"], **common),
        ),
        Chunk(
            level=ChunkLevel.CHILD,
            text=_TEXT,
            metadata=ChunkMetadata(chunk_id=child_id, parent_id=parent_id, tags=["bug"], **common),
        ),
    ]


def _doc_unit(source_id: str, parent_index: int, section: str) -> list[Chunk]:
    parent_id = doc_parent_id(source_id, parent_index)
    child_id = doc_child_id(source_id, parent_index, 0)
    common = {
        "source_type": ChunkSourceType.DOCS,
        "source_id": source_id,
        "title": "Doc",
        "section_path": ["Top", section],
        "airflow_area": "core_concepts",
    }
    return [
        Chunk(
            level=ChunkLevel.PARENT,
            text=_TEXT,
            metadata=ChunkMetadata(
                chunk_id=parent_id, parent_id=None, tags=["core_concepts"], **common
            ),
        ),
        Chunk(
            level=ChunkLevel.CHILD,
            text=_TEXT,
            metadata=ChunkMetadata(
                chunk_id=child_id, parent_id=parent_id, tags=["core_concepts"], **common
            ),
        ),
    ]


def _chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(20):  # >= 8 issue_only + 12 mixed
        chunks += _issue_unit(github_id=1000 + i, comment_id=50 + i)
    for j in range(6):  # >= 5 docs_only
        chunks += _doc_unit(source_id=f"doc_{j}", parent_index=0, section=f"Section{j}")
    return chunks


def test_build_draft_golden_distribution_and_grounding() -> None:
    chunks = _chunks()
    valid_ids = {c.chunk_id for c in chunks}
    examples = build_draft_golden(chunks)

    assert len(examples) == GOLDEN_SIZE == 25
    distribution = Counter(e.source_type for e in examples)
    assert distribution == EXPECTED_DISTRIBUTION

    for example in examples:
        assert example.human_reviewed is False
        assert example.ground_truth_child_chunk_ids
        for chunk_id in (*example.ground_truth_parent_ids, *example.ground_truth_child_chunk_ids):
            assert chunk_id in valid_ids


def test_validate_golden_accepts_valid_draft() -> None:
    chunks = _chunks()
    valid_ids = {c.chunk_id for c in chunks}
    examples = build_draft_golden(chunks)
    assert validate_golden_examples(examples, valid_ids) == []


def test_validate_golden_flags_unknown_chunk_and_distribution() -> None:
    chunks = _chunks()
    valid_ids = {c.chunk_id for c in chunks}
    examples = build_draft_golden(chunks)
    # Corrupt one example's chunk reference and drop one to break the distribution.
    examples[0].ground_truth_child_chunk_ids = ["doc:nope:parent:0:child:0"]
    errors = validate_golden_examples(examples[:-1], valid_ids)
    assert any("unknown chunk id" in e for e in errors)
    assert any("expected 25 examples" in e for e in errors)


def test_validate_golden_require_human_reviewed_freeze_gate() -> None:
    chunks = _chunks()
    valid_ids = {c.chunk_id for c in chunks}
    examples = build_draft_golden(chunks)
    errors = validate_golden_examples(examples, valid_ids, require_human_reviewed=True)
    assert any("not human_reviewed" in e for e in errors)


def test_build_draft_golden_requires_enough_units() -> None:
    with pytest.raises(ValueError, match="Not enough grounded units"):
        build_draft_golden(_issue_unit(1, 2))
