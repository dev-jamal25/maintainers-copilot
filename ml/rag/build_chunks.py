"""Build the RAG chunk artifacts from the normalized corpora (A06, DECISIONS D3.2).

Reads the docs corpus (``rag_doc_corpus.jsonl``, A03) and issue corpus
(``rag_issue_corpus.jsonl``, A04), then writes two deterministic artifacts:

- ``data/processed/rag_chunks.jsonl`` -- advanced parent-child chunks (the strategy to defend).
- ``data/processed/rag_chunks_naive.jsonl`` -- flat fixed-size chunks (the baseline to beat).

It validates unique chunk IDs and parent referential integrity, and records the active
tokenizer + chunking params in ``data/processed/rag_corpus_manifest.json`` for reproducibility.

Run (cwd = ml so the ``rag`` package resolves):
    uv run --directory ml python -m rag.build_chunks
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.domain.rag import Chunk, ChunkLevel, CorpusRecord

from rag.chunking import (
    DEFAULT_PARAMS,
    ChunkingParams,
    TokenCounter,
    active_tokenizer_name,
    chunk_corpus_record,
    default_token_counter,
    naive_chunk_corpus_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_CORPUS = REPO_ROOT / "data" / "processed" / "rag_doc_corpus.jsonl"
DEFAULT_ISSUE_CORPUS = REPO_ROOT / "data" / "processed" / "rag_issue_corpus.jsonl"
DEFAULT_ADVANCED_OUT = REPO_ROOT / "data" / "processed" / "rag_chunks.jsonl"
DEFAULT_NAIVE_OUT = REPO_ROOT / "data" / "processed" / "rag_chunks_naive.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "rag_corpus_manifest.json"


def load_corpus_records(path: Path) -> list[CorpusRecord]:
    """Load and validate CorpusRecord rows from a corpus JSONL (skips a missing file)."""
    if not path.exists():
        return []
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(CorpusRecord.model_validate_json(line))
    return records


def build_advanced_chunks(
    records: list[CorpusRecord],
    params: ChunkingParams,
    count: TokenCounter = default_token_counter,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(chunk_corpus_record(record, params, count))
    return chunks


def build_naive_chunks(
    records: list[CorpusRecord],
    params: ChunkingParams,
    count: TokenCounter = default_token_counter,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(naive_chunk_corpus_record(record, params, count))
    return chunks


def assert_unique_ids(chunks: list[Chunk], *, label: str) -> None:
    counts = Counter(chunk.chunk_id for chunk in chunks)
    dupes = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"Duplicate {label} chunk IDs: {dupes[:10]}")


def assert_parents_exist(chunks: list[Chunk]) -> None:
    """Every child's parent_id must resolve to a parent chunk in the same set."""
    parent_ids = {c.chunk_id for c in chunks if c.level == ChunkLevel.PARENT}
    missing = sorted(
        {
            c.parent_id
            for c in chunks
            if c.level == ChunkLevel.CHILD
            and c.parent_id is not None
            and c.parent_id not in parent_ids
        }
    )
    if missing:
        raise ValueError(f"Child chunks reference missing parents: {missing[:10]}")


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            handle.write(payload + "\n")


def _level_counts(chunks: list[Chunk]) -> dict[str, int]:
    parents = sum(1 for c in chunks if c.level == ChunkLevel.PARENT)
    return {"total": len(chunks), "parents": parents, "children": len(chunks) - parents}


def build_manifest(
    doc_records: list[CorpusRecord],
    issue_records: list[CorpusRecord],
    advanced: list[Chunk],
    naive: list[Chunk],
    params: ChunkingParams,
    *,
    tokenizer: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "tokenizer": tokenizer,
        "chunking_params": {
            "max_child_tokens": params.max_child_tokens,
            "min_child_tokens": params.min_child_tokens,
            "overlap_tokens": params.overlap_tokens,
            "naive_chunk_tokens": params.naive_chunk_tokens,
        },
        "doc_records": len(doc_records),
        "issue_records": len(issue_records),
        "advanced": _level_counts(advanced),
        "naive": _level_counts(naive),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG chunk artifacts from the corpora.")
    parser.add_argument("--doc-corpus", type=Path, default=DEFAULT_DOC_CORPUS)
    parser.add_argument("--issue-corpus", type=Path, default=DEFAULT_ISSUE_CORPUS)
    parser.add_argument("--advanced-out", type=Path, default=DEFAULT_ADVANCED_OUT)
    parser.add_argument("--naive-out", type=Path, default=DEFAULT_NAIVE_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = DEFAULT_PARAMS
    doc_records = load_corpus_records(args.doc_corpus)
    issue_records = load_corpus_records(args.issue_corpus)
    records = doc_records + issue_records

    advanced = build_advanced_chunks(records, params)
    naive = build_naive_chunks(records, params)
    assert_unique_ids(advanced, label="advanced")
    assert_unique_ids(naive, label="naive")
    assert_parents_exist(advanced)

    write_chunks_jsonl(advanced, args.advanced_out)
    write_chunks_jsonl(naive, args.naive_out)
    manifest = build_manifest(
        doc_records,
        issue_records,
        advanced,
        naive,
        params,
        tokenizer=active_tokenizer_name(),
        generated_at=datetime.now(UTC).isoformat(),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"tokenizer: {manifest['tokenizer']}")
    print(f"doc records: {len(doc_records)} | issue records: {len(issue_records)}")
    print(f"advanced chunks: {manifest['advanced']}")
    print(f"naive chunks: {manifest['naive']}")
    print(f"outputs: {args.advanced_out.name}, {args.naive_out.name}, {args.manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
