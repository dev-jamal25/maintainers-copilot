"""Hierarchical parent-child chunking + naive fixed-size baseline (A05, DECISIONS D3.2).

Advanced strategy (the non-naive chunking that beats the baseline):
- Docs: split a normalized page into per-heading sections (parents, with a breadcrumb
  ``section_path``); pack each parent's blocks into children of <= ``max_child_tokens`` with
  ``overlap_tokens`` carry-over. We embed/retrieve children and expand to parents for context.
- Issues: each CorpusRecord (issue body or one maintainer comment) is a single parent
  (comment-scoped IDs, D3.7); pack it into children the same way.

Naive baseline: fixed-size token windows within each source, no hierarchy (flat children).

Token counting uses the bge-small tokenizer when available and a deterministic word-count
fallback otherwise, so the chunker runs offline (e.g. CI) and in tests. The active counter is
recorded by the artifact builder (A06) for reproducibility.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from backend.app.domain.rag import (
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkSourceType,
    CorpusRecord,
    doc_child_id,
    doc_naive_id,
    doc_parent_id,
    issue_child_id,
    issue_naive_id,
    issue_parent_id,
)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HEADING_RE = re.compile(r"^(#+)\s+(.*)$")

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ChunkingParams:
    """Chunking knobs (D3.2 defaults: child 350-500 tokens, overlap 50-75)."""

    max_child_tokens: int = 500
    min_child_tokens: int = 350
    overlap_tokens: int = 64
    naive_chunk_tokens: int = 400


DEFAULT_PARAMS = ChunkingParams()

_tokenizer: object | None = None
_tokenizer_loaded = False


def _load_tokenizer() -> object | None:
    """Lazily load the bge tokenizer; return None (word-count fallback) if unavailable."""
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer
    _tokenizer_loaded = True
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    except Exception:  # noqa: BLE001 - any load failure (offline, missing files) -> deterministic fallback
        _tokenizer = None
    return _tokenizer


def default_token_counter(text: str) -> int:
    """Count tokens with the bge tokenizer, falling back to whitespace word count."""
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]


def active_tokenizer_name() -> str:
    """Name of the token counter actually in use (for the artifact manifest)."""
    return EMBED_MODEL_NAME if _load_tokenizer() is not None else "wordcount-fallback"


def split_words(text: str) -> list[str]:
    """Whitespace word atoms (used by the naive baseline)."""
    return text.split()


def split_blocks(text: str) -> list[str]:
    """Split into atomic blocks: fenced code blocks stay whole; prose splits on blank lines."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        if buffer:
            joined = "\n".join(buffer).strip("\n")
            if joined.strip():
                blocks.append(joined)

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                buffer.append(line)
                flush()
                buffer.clear()
                in_code = False
            else:
                flush()
                buffer.clear()
                in_code = True
                buffer.append(line)
            continue
        if in_code:
            buffer.append(line)
        elif not line.strip():
            flush()
            buffer.clear()
        else:
            buffer.append(line)
    flush()
    return blocks


def split_doc_sections(text: str) -> list[tuple[list[str], str]]:
    """Split a normalized doc into (section_path, body) sections by ``#`` headings.

    A section runs from a heading to the next heading of equal-or-higher level; ``section_path``
    is the breadcrumb of ancestor headings. Preamble before the first heading is its own section.
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    cur_path: list[str] = []
    cur_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((list(cur_path), body))

    for line in text.split("\n"):
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = [item[1] for item in stack]
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush()
    return sections


def _split_oversize(segment: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split a single over-budget segment into word windows of <= max_tokens."""
    words = segment.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(
    segments: list[str], overlap_tokens: int, count: TokenCounter
) -> tuple[list[str], int]:
    """Return trailing segments (and their token sum) to carry into the next child as overlap."""
    if overlap_tokens <= 0:
        return [], 0
    tail: list[str] = []
    total = 0
    for segment in reversed(segments):
        tokens = count(segment)
        if tail and total + tokens > overlap_tokens:
            break
        tail.insert(0, segment)
        total += tokens
        if total >= overlap_tokens:
            break
    return tail, total


def pack_segments(
    segments: list[str],
    *,
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
    joiner: str,
) -> list[str]:
    """Greedily pack atomic segments into chunks of <= max_tokens with token overlap."""
    expanded: list[str] = []
    for segment in segments:
        if count(segment) > max_tokens:
            expanded.extend(_split_oversize(segment, max_tokens, count))
        else:
            expanded.append(segment)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for segment in expanded:
        tokens = count(segment)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(joiner.join(current))
            tail, tail_tokens = _overlap_tail(current, overlap_tokens, count)
            # Overlap must never block progress: if the carried tail plus this segment would
            # still overflow, start the next chunk fresh instead of looping forever.
            if tail_tokens + tokens > max_tokens:
                tail, tail_tokens = [], 0
            current, current_tokens = tail, tail_tokens
        current.append(segment)
        current_tokens += tokens
    if current:
        chunks.append(joiner.join(current))
    return chunks


def _doc_metadata(
    record: CorpusRecord, chunk_id: str, parent_id: str | None, section_path: list[str]
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        parent_id=parent_id,
        source_type=ChunkSourceType.DOCS,
        source_id=record.source_id,
        title=record.title,
        url=record.url,
        section_path=list(section_path),
        airflow_area=record.airflow_area,
        tags=list(record.tags),
        version=record.version,
        github_issue_id=None,
        github_comment_id=None,
        created_at=record.created_at,
        closed_at=record.closed_at,
    )


def _issue_metadata(record: CorpusRecord, chunk_id: str, parent_id: str | None) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        parent_id=parent_id,
        source_type=ChunkSourceType.ISSUE,
        source_id=record.source_id,
        title=record.title,
        url=record.url,
        section_path=[],
        airflow_area=record.airflow_area,
        tags=list(record.tags),
        version=record.version,
        github_issue_id=record.github_issue_id,
        github_comment_id=record.github_comment_id,
        created_at=record.created_at,
        closed_at=record.closed_at,
    )


def _chunk_doc(record: CorpusRecord, params: ChunkingParams, count: TokenCounter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for parent_index, (section_path, body) in enumerate(split_doc_sections(record.text)):
        parent_id = doc_parent_id(record.source_id, parent_index)
        chunks.append(
            Chunk(
                level=ChunkLevel.PARENT,
                text=body,
                metadata=_doc_metadata(record, parent_id, None, section_path),
            )
        )
        child_texts = pack_segments(
            split_blocks(body),
            max_tokens=params.max_child_tokens,
            overlap_tokens=params.overlap_tokens,
            count=count,
            joiner="\n\n",
        )
        for child_index, child_text in enumerate(child_texts):
            child_id = doc_child_id(record.source_id, parent_index, child_index)
            chunks.append(
                Chunk(
                    level=ChunkLevel.CHILD,
                    text=child_text,
                    metadata=_doc_metadata(record, child_id, parent_id, section_path),
                )
            )
    return chunks


def _chunk_issue(record: CorpusRecord, params: ChunkingParams, count: TokenCounter) -> list[Chunk]:
    if record.github_issue_id is None or record.github_comment_id is None:
        raise ValueError(f"Issue CorpusRecord requires github ids: {record.source_id}")
    github_id = record.github_issue_id
    comment_id = record.github_comment_id
    parent_id = issue_parent_id(github_id, comment_id, 0)
    chunks: list[Chunk] = [
        Chunk(
            level=ChunkLevel.PARENT,
            text=record.text,
            metadata=_issue_metadata(record, parent_id, None),
        )
    ]
    child_texts = pack_segments(
        split_blocks(record.text),
        max_tokens=params.max_child_tokens,
        overlap_tokens=params.overlap_tokens,
        count=count,
        joiner="\n\n",
    )
    for child_index, child_text in enumerate(child_texts):
        child_id = issue_child_id(github_id, comment_id, child_index)
        chunks.append(
            Chunk(
                level=ChunkLevel.CHILD,
                text=child_text,
                metadata=_issue_metadata(record, child_id, parent_id),
            )
        )
    return chunks


def chunk_corpus_record(
    record: CorpusRecord,
    params: ChunkingParams = DEFAULT_PARAMS,
    count: TokenCounter = default_token_counter,
) -> list[Chunk]:
    """Parent-child chunk a single corpus record (advanced strategy)."""
    if record.source_type == ChunkSourceType.DOCS:
        return _chunk_doc(record, params, count)
    return _chunk_issue(record, params, count)


def naive_chunk_corpus_record(
    record: CorpusRecord,
    params: ChunkingParams = DEFAULT_PARAMS,
    count: TokenCounter = default_token_counter,
) -> list[Chunk]:
    """Flat fixed-size chunks within a source (the baseline to beat); no parent-child."""
    pieces = pack_segments(
        split_words(record.text),
        max_tokens=params.naive_chunk_tokens,
        overlap_tokens=0,
        count=count,
        joiner=" ",
    )
    chunks: list[Chunk] = []
    for index, text in enumerate(pieces):
        if record.source_type == ChunkSourceType.DOCS:
            chunk_id = doc_naive_id(record.source_id, index)
            metadata = _doc_metadata(record, chunk_id, None, [])
        else:
            if record.github_issue_id is None or record.github_comment_id is None:
                raise ValueError(f"Issue CorpusRecord requires github ids: {record.source_id}")
            chunk_id = issue_naive_id(record.github_issue_id, record.github_comment_id, index)
            metadata = _issue_metadata(record, chunk_id, None)
        chunks.append(Chunk(level=ChunkLevel.CHILD, text=text, metadata=metadata))
    return chunks
