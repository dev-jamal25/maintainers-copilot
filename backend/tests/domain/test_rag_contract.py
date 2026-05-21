from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.rag import (
    ISSUE_BODY_COMMENT_ID,
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkSourceType,
    Difficulty,
    GoldenExample,
    GoldenSourceType,
    SourceRef,
    doc_child_id,
    doc_parent_id,
    is_child_id,
    issue_child_id,
    issue_parent_id,
)


def test_doc_id_helpers_match_frozen_patterns() -> None:
    assert doc_parent_id("scheduler_overview", 0) == "doc:scheduler_overview:parent:0"
    assert doc_child_id("scheduler_overview", 0, 2) == "doc:scheduler_overview:parent:0:child:2"


def test_issue_id_helpers_match_frozen_patterns() -> None:
    assert issue_parent_id(63532, 998877, 0) == "issue:63532:comment:998877:parent:0"
    # Frozen issue child pattern omits the parent index.
    assert issue_child_id(63532, 998877, 1) == "issue:63532:comment:998877:child:1"
    # Issue body uses the sentinel comment id.
    assert issue_parent_id(63532, ISSUE_BODY_COMMENT_ID, 0) == "issue:63532:comment:0:parent:0"


def test_is_child_id() -> None:
    assert is_child_id(doc_child_id("d", 0, 0))
    assert is_child_id(issue_child_id(1, 2, 0))
    assert not is_child_id(doc_parent_id("d", 0))
    assert not is_child_id(issue_parent_id(1, 2, 0))


def test_chunk_metadata_roundtrips() -> None:
    meta = ChunkMetadata(
        chunk_id=doc_child_id("dags_howto", 1, 0),
        parent_id=doc_parent_id("dags_howto", 1),
        source_type=ChunkSourceType.DOCS,
        source_id="dags_howto",
        title="Writing DAGs",
        section_path=["Core Concepts", "DAGs"],
        airflow_area="dags",
        tags=["dags", "howto"],
    )
    reloaded = ChunkMetadata.model_validate_json(meta.model_dump_json())
    assert reloaded == meta


def test_chunk_metadata_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata.model_validate(
            {
                "chunk_id": "doc:x:parent:0",
                "source_type": "docs",
                "source_id": "x",
                "unexpected": "field",
            }
        )


def test_chunk_level_and_child_flag() -> None:
    parent = Chunk(
        level=ChunkLevel.PARENT,
        text="section text",
        metadata=ChunkMetadata(
            chunk_id=doc_parent_id("x", 0),
            source_type=ChunkSourceType.DOCS,
            source_id="x",
        ),
    )
    child = Chunk(
        level=ChunkLevel.CHILD,
        text="child text",
        metadata=ChunkMetadata(
            chunk_id=doc_child_id("x", 0, 0),
            parent_id=doc_parent_id("x", 0),
            source_type=ChunkSourceType.DOCS,
            source_id="x",
        ),
    )
    assert not parent.is_child
    assert parent.parent_id is None
    assert child.is_child
    assert child.parent_id == doc_parent_id("x", 0)
    assert child.chunk_id == doc_child_id("x", 0, 0)


def test_golden_example_defaults_are_draft() -> None:
    example = GoldenExample(
        question="How do I fix a stuck scheduler?",
        ideal_answer="Restart the scheduler after clearing the stale job rows ...",
        ground_truth_sources=[
            SourceRef(
                source_type=ChunkSourceType.ISSUE,
                source_id="issue:63532",
                github_issue_id=63532,
                github_comment_id=998877,
                quote="clear the stale job rows",
            )
        ],
        difficulty=Difficulty.MEDIUM,
        source_type=GoldenSourceType.ISSUE_ONLY,
    )
    assert example.human_reviewed is False
    assert example.ground_truth_parent_ids == []
    assert example.ground_truth_child_chunk_ids == []


def test_enum_string_values_are_frozen() -> None:
    assert ChunkSourceType.DOCS.value == "docs"
    assert ChunkSourceType.ISSUE.value == "issue"
    assert GoldenSourceType.MIXED.value == "mixed"
    assert Difficulty.HARD.value == "hard"
    assert ChunkLevel.CHILD.value == "child"
