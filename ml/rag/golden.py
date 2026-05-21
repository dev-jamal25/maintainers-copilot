"""RAG golden-set draft generator + validator (A07, DECISIONS D3.4).

Builds 25 *draft* golden examples grounded in the real chunk artifact (no Airflow facts written
from memory): questions are derived from issue titles / doc section headings, ideal answers are
extracted verbatim from maintainer comments / doc sections, and every example resolves to real
parent + child chunk IDs. Drafts are marked ``human_reviewed = False``.

Distribution (D3.4): 8 issue_only, 5 docs_only, 12 mixed.

These drafts are a starting point only. A maintainer must confirm/refine the grounded ideal
answers and hand-label 5/25 (D1.12) before freezing ``data/evals/rag_golden.jsonl``. The
validator enforces distribution, schema, and referential integrity against the chunk artifact;
``require_human_reviewed`` additionally enforces the freeze gate for the final set.

Run (cwd = ml so the ``rag`` package resolves):
    uv run --directory ml python -m rag.golden
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.domain.rag import (
    Chunk,
    ChunkLevel,
    ChunkSourceType,
    Difficulty,
    GoldenExample,
    GoldenSourceType,
    SourceRef,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = REPO_ROOT / "data" / "processed" / "rag_chunks.jsonl"
DEFAULT_DRAFT_OUT = REPO_ROOT / "data" / "evals" / "rag_golden.draft.jsonl"

EXPECTED_DISTRIBUTION: dict[GoldenSourceType, int] = {
    GoldenSourceType.ISSUE_ONLY: 8,
    GoldenSourceType.DOCS_ONLY: 5,
    GoldenSourceType.MIXED: 12,
}
GOLDEN_SIZE = sum(EXPECTED_DISTRIBUTION.values())

_ANSWER_MAX_CHARS = 1200
_QUOTE_MAX_CHARS = 160
_MIN_ANSWER_CHARS = 80
_DIFFICULTY_CYCLE = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)


@dataclass(frozen=True)
class SectionUnit:
    """A parent chunk plus its child chunk IDs (a retrievable section/comment)."""

    parent: Chunk
    child_ids: list[str]


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def _children_by_parent(chunks: list[Chunk]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk.level == ChunkLevel.CHILD and chunk.parent_id is not None:
            mapping[chunk.parent_id].append(chunk.chunk_id)
    return mapping


def _units(chunks: list[Chunk], predicate: Callable[[Chunk], bool]) -> list[SectionUnit]:
    children = _children_by_parent(chunks)
    units = [
        SectionUnit(parent=chunk, child_ids=sorted(children.get(chunk.chunk_id, [])))
        for chunk in chunks
        if chunk.level == ChunkLevel.PARENT and predicate(chunk) and children.get(chunk.chunk_id)
    ]
    units.sort(key=lambda unit: unit.parent.chunk_id)
    return units


def issue_comment_units(chunks: list[Chunk]) -> list[SectionUnit]:
    """Maintainer-comment parents (comment id != body sentinel) with enough text to answer."""
    return _units(
        chunks,
        lambda c: (
            c.metadata.source_type == ChunkSourceType.ISSUE
            and c.metadata.github_comment_id not in (None, 0)
            and len(c.text) >= _MIN_ANSWER_CHARS
        ),
    )


def doc_section_units(chunks: list[Chunk]) -> list[SectionUnit]:
    """Doc section parents with a breadcrumb and enough text to answer."""
    return _units(
        chunks,
        lambda c: (
            c.metadata.source_type == ChunkSourceType.DOCS
            and bool(c.metadata.section_path)
            and len(c.text) >= _MIN_ANSWER_CHARS
        ),
    )


def _distinct_issue_units(units: list[SectionUnit]) -> list[SectionUnit]:
    """Keep one comment per distinct issue + distinct title (diverse, non-duplicate questions)."""
    seen_issues: set[int | None] = set()
    seen_titles: set[str] = set()
    distinct: list[SectionUnit] = []
    for unit in units:
        github_id = unit.parent.metadata.github_issue_id
        title = (unit.parent.metadata.title or "").strip()
        if github_id in seen_issues or title in seen_titles:
            continue
        seen_issues.add(github_id)
        seen_titles.add(title)
        distinct.append(unit)
    return distinct


def _distinct_doc_units(units: list[SectionUnit]) -> list[SectionUnit]:
    """Keep one section per distinct doc + distinct section title."""
    seen_sources: set[str] = set()
    seen_titles: set[str] = set()
    distinct: list[SectionUnit] = []
    for unit in units:
        source_id = unit.parent.metadata.source_id
        path = unit.parent.metadata.section_path
        title = path[-1] if path else source_id
        if source_id in seen_sources or title in seen_titles:
            continue
        seen_sources.add(source_id)
        seen_titles.add(title)
        distinct.append(unit)
    return distinct


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    return (cut[:last_space] if last_space > 0 else cut).rstrip() + " ..."


def _difficulty(index: int) -> Difficulty:
    return _DIFFICULTY_CYCLE[index % len(_DIFFICULTY_CYCLE)]


def _issue_question(unit: SectionUnit) -> str:
    title = (unit.parent.metadata.title or "").strip()
    return title or f"How do maintainers resolve issue {unit.parent.metadata.github_issue_id}?"


def _issue_source(unit: SectionUnit) -> SourceRef:
    meta = unit.parent.metadata
    return SourceRef(
        source_type=ChunkSourceType.ISSUE,
        source_id=meta.source_id,
        github_issue_id=meta.github_issue_id,
        github_comment_id=meta.github_comment_id,
        quote=_trim(unit.parent.text, _QUOTE_MAX_CHARS),
    )


def _doc_source(unit: SectionUnit) -> SourceRef:
    meta = unit.parent.metadata
    return SourceRef(
        source_type=ChunkSourceType.DOCS,
        source_id=meta.source_id,
        section_path=list(meta.section_path),
        quote=_trim(unit.parent.text, _QUOTE_MAX_CHARS),
    )


def build_draft_golden(chunks: list[Chunk]) -> list[GoldenExample]:
    """Assemble 25 grounded draft examples (8 issue_only, 5 docs_only, 12 mixed)."""
    issues = _distinct_issue_units(issue_comment_units(chunks))
    docs = _distinct_doc_units(doc_section_units(chunks))
    n_issue_only = EXPECTED_DISTRIBUTION[GoldenSourceType.ISSUE_ONLY]
    n_docs_only = EXPECTED_DISTRIBUTION[GoldenSourceType.DOCS_ONLY]
    n_mixed = EXPECTED_DISTRIBUTION[GoldenSourceType.MIXED]
    if len(issues) < n_issue_only + n_mixed or len(docs) < max(n_docs_only, 1):
        raise ValueError("Not enough grounded units in the chunk artifact to build the golden set")

    examples: list[GoldenExample] = []

    for index, unit in enumerate(issues[:n_issue_only]):
        meta = unit.parent.metadata
        examples.append(
            GoldenExample(
                question=_issue_question(unit),
                ideal_answer=_trim(unit.parent.text, _ANSWER_MAX_CHARS),
                ground_truth_sources=[_issue_source(unit)],
                tags=list(meta.tags),
                difficulty=_difficulty(index),
                source_type=GoldenSourceType.ISSUE_ONLY,
                ground_truth_parent_ids=[unit.parent.chunk_id],
                ground_truth_child_chunk_ids=list(unit.child_ids),
                human_reviewed=False,
            )
        )

    for index, unit in enumerate(docs[:n_docs_only]):
        meta = unit.parent.metadata
        if meta.section_path:
            section_title = meta.section_path[-1]
        else:
            section_title = meta.title or meta.source_id
        examples.append(
            GoldenExample(
                question=f"What do the Airflow docs say about {section_title}?",
                ideal_answer=_trim(unit.parent.text, _ANSWER_MAX_CHARS),
                ground_truth_sources=[_doc_source(unit)],
                tags=list(meta.tags),
                difficulty=_difficulty(index),
                source_type=GoldenSourceType.DOCS_ONLY,
                ground_truth_parent_ids=[unit.parent.chunk_id],
                ground_truth_child_chunk_ids=list(unit.child_ids),
                human_reviewed=False,
            )
        )

    mixed_issues = issues[n_issue_only : n_issue_only + n_mixed]
    for index, issue_unit in enumerate(mixed_issues):
        doc_unit = docs[index % len(docs)]
        issue_meta = issue_unit.parent.metadata
        doc_meta = doc_unit.parent.metadata
        ideal = (
            f"{_trim(issue_unit.parent.text, _ANSWER_MAX_CHARS // 2)}\n\n"
            f"{_trim(doc_unit.parent.text, _ANSWER_MAX_CHARS // 2)}"
        )
        examples.append(
            GoldenExample(
                question=_issue_question(issue_unit),
                ideal_answer=ideal,
                ground_truth_sources=[_issue_source(issue_unit), _doc_source(doc_unit)],
                tags=sorted({*issue_meta.tags, *doc_meta.tags}),
                difficulty=_difficulty(index),
                source_type=GoldenSourceType.MIXED,
                ground_truth_parent_ids=[issue_unit.parent.chunk_id, doc_unit.parent.chunk_id],
                ground_truth_child_chunk_ids=[*issue_unit.child_ids, *doc_unit.child_ids],
                human_reviewed=False,
            )
        )

    return examples


def validate_golden_examples(
    examples: list[GoldenExample],
    valid_chunk_ids: set[str],
    *,
    require_human_reviewed: bool = False,
) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []
    if len(examples) != GOLDEN_SIZE:
        errors.append(f"expected {GOLDEN_SIZE} examples, found {len(examples)}")

    distribution = Counter(example.source_type for example in examples)
    for source_type, expected in EXPECTED_DISTRIBUTION.items():
        if distribution.get(source_type, 0) != expected:
            errors.append(
                f"distribution for {source_type}: expected {expected}, "
                f"found {distribution.get(source_type, 0)}"
            )

    seen_questions: set[str] = set()
    for index, example in enumerate(examples):
        if not example.question.strip():
            errors.append(f"[{index}] empty question")
        if len(example.ideal_answer.strip()) < _MIN_ANSWER_CHARS:
            errors.append(f"[{index}] ideal_answer too short to be grounded")
        if not example.ground_truth_sources:
            errors.append(f"[{index}] no ground_truth_sources")
        if not example.ground_truth_child_chunk_ids:
            errors.append(f"[{index}] no ground_truth_child_chunk_ids")
        for chunk_id in (*example.ground_truth_parent_ids, *example.ground_truth_child_chunk_ids):
            if chunk_id not in valid_chunk_ids:
                errors.append(f"[{index}] unknown chunk id: {chunk_id}")
        if example.question in seen_questions:
            errors.append(f"[{index}] duplicate question: {example.question[:60]}")
        seen_questions.add(example.question)
        if require_human_reviewed and not example.human_reviewed:
            errors.append(f"[{index}] not human_reviewed (freeze gate)")
    return errors


def write_golden_jsonl(examples: list[GoldenExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = json.dumps(
                example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            handle.write(payload + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build + validate the RAG golden-set draft.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_DRAFT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)
    valid_ids = {chunk.chunk_id for chunk in chunks}
    examples = build_draft_golden(chunks)
    errors = validate_golden_examples(examples, valid_ids)
    if errors:
        for error in errors:
            print(f"INVALID: {error}")
        raise SystemExit(1)
    write_golden_jsonl(examples, args.output)
    distribution = Counter(example.source_type.value for example in examples)
    print(f"golden draft examples: {len(examples)} {dict(distribution)}")
    print(f"output: {args.output}")
    print("NOTE: drafts are human_reviewed=false. Refine answers + hand-label 5/25, then")
    print("      freeze data/evals/rag_golden.jsonl (validate with require_human_reviewed=True).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
