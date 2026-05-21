"""Canonical RAG domain contract for Day 3.

Single source of truth for chunk metadata, parent/child chunks, source references,
golden-set examples, retrieval results, eval-report shapes, and the stable chunk-ID
helpers (DECISIONS D3.4, D3.6, D3.7).

Import-safety: this module imports only ``pydantic`` + the standard library so it stays
importable from every sub-project environment (backend, ml, scripts, evals) without
pulling backend-only or heavy dependencies. Do not add SQLAlchemy/httpx/settings imports
here.

Chunk-ID patterns (frozen, D3.7) — formatted only via the helpers below:
    doc:{source_id}:parent:{n}
    doc:{source_id}:parent:{n}:child:{m}
    issue:{github_id}:comment:{comment_id}:parent:{n}
    issue:{github_id}:comment:{comment_id}:child:{m}

Note the frozen issue *child* pattern omits the parent index. To keep child IDs unique we
therefore enforce exactly one parent per issue source block (a single comment or the issue
body); the issue body uses the sentinel comment id ``0``. Doc child IDs embed the parent
index, so docs may have multiple parents per source.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# --- Sentinels ---------------------------------------------------------------

#: Sentinel comment id standing in for the issue body/opening (issues are comment-scoped
#: in the frozen D3.7 ID scheme; real comment ids are GitHub comment ``github_id`` values).
ISSUE_BODY_COMMENT_ID = 0


# --- Enums -------------------------------------------------------------------


class ChunkSourceType(StrEnum):
    """Where a chunk came from (D3.6 ``source_type``)."""

    DOCS = "docs"
    ISSUE = "issue"


class ChunkLevel(StrEnum):
    """Parent (context) vs child (retrievable) chunk."""

    PARENT = "parent"
    CHILD = "child"


class GoldenSourceType(StrEnum):
    """Golden-example grounding mix (D3.4 ``source_type``)."""

    ISSUE_ONLY = "issue_only"
    DOCS_ONLY = "docs_only"
    MIXED = "mixed"


class Difficulty(StrEnum):
    """Golden-example difficulty (D3.4)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# --- Chunk-ID helpers (the only place IDs are formatted) ---------------------


def doc_parent_id(source_id: str, parent_index: int) -> str:
    """Return the stable ID for a documentation parent chunk."""
    return f"doc:{source_id}:parent:{parent_index}"


def doc_child_id(source_id: str, parent_index: int, child_index: int) -> str:
    """Return the stable ID for a documentation child chunk."""
    return f"doc:{source_id}:parent:{parent_index}:child:{child_index}"


def issue_parent_id(github_id: int, comment_id: int, parent_index: int) -> str:
    """Return the stable ID for an issue/comment parent chunk."""
    return f"issue:{github_id}:comment:{comment_id}:parent:{parent_index}"


def issue_child_id(github_id: int, comment_id: int, child_index: int) -> str:
    """Return the stable ID for an issue/comment child chunk (frozen pattern omits parent)."""
    return f"issue:{github_id}:comment:{comment_id}:child:{child_index}"


def doc_naive_id(source_id: str, index: int) -> str:
    """Return the ID for a doc chunk in the naive fixed-size baseline (flat, no hierarchy)."""
    return f"doc:{source_id}:naive:{index}"


def issue_naive_id(github_id: int, comment_id: int, index: int) -> str:
    """Return the ID for an issue chunk in the naive fixed-size baseline (flat, no hierarchy)."""
    return f"issue:{github_id}:comment:{comment_id}:naive:{index}"


def is_child_id(chunk_id: str) -> bool:
    """True when ``chunk_id`` names a child (retrievable) chunk."""
    return ":child:" in chunk_id


# --- Chunk models ------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Minimal stable metadata attached to every chunk before indexing (frozen, D3.6).

    Field set is exactly the frozen D3.6 schema; ``extra="forbid"`` guards against drift.
    ``parent_id`` is ``None`` for parent chunks and the parent's ``chunk_id`` for children.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_id: str | None = None
    source_type: ChunkSourceType
    source_id: str
    title: str | None = None
    url: str | None = None
    section_path: list[str] = Field(default_factory=list)
    airflow_area: str | None = None
    tags: list[str] = Field(default_factory=list)
    version: str | None = None
    github_issue_id: int | None = None
    github_comment_id: int | None = None
    created_at: datetime | None = None
    closed_at: datetime | None = None


class Chunk(BaseModel):
    """A single chunk (parent or child): its level, text, and frozen metadata.

    One canonical chunk model — there is intentionally no separate ParentChunk/ChildChunk
    type; ``level`` plus ``metadata.parent_id`` carry the distinction.
    """

    model_config = ConfigDict(extra="forbid")

    level: ChunkLevel
    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def parent_id(self) -> str | None:
        return self.metadata.parent_id

    @property
    def is_child(self) -> bool:
        return self.level is ChunkLevel.CHILD


class CorpusRecord(BaseModel):
    """A normalized pre-chunk source record (the scripts -> ml handoff).

    Docs scripts (A03) and the holdout-issue script (A04) emit these as JSONL; the chunk
    builder (A06) validates and chunks them. A docs record carries a whole normalized page
    (chunked into per-section parents); an issue record carries one comment or the issue body
    (a single parent), so ``github_comment_id`` is set (``ISSUE_BODY_COMMENT_ID`` for the body).
    """

    model_config = ConfigDict(extra="forbid")

    source_type: ChunkSourceType
    source_id: str
    text: str
    title: str | None = None
    url: str | None = None
    airflow_area: str | None = None
    tags: list[str] = Field(default_factory=list)
    version: str | None = None
    github_issue_id: int | None = None
    github_comment_id: int | None = None
    created_at: datetime | None = None
    closed_at: datetime | None = None


# --- Golden set --------------------------------------------------------------


class SourceRef(BaseModel):
    """A reference to a grounding source, used in golden drafts before ID resolution."""

    model_config = ConfigDict(extra="forbid")

    source_type: ChunkSourceType
    source_id: str
    github_issue_id: int | None = None
    github_comment_id: int | None = None
    section_path: list[str] = Field(default_factory=list)
    quote: str | None = None
    note: str | None = None


class GoldenExample(BaseModel):
    """A RAG golden triple (D3.4): draft fields plus post-ingestion resolved chunk IDs.

    ``human_reviewed`` stays ``False`` until a maintainer confirms the grounded ideal answer
    and (for the 5/25 honesty subset) hand-labels it; the set is frozen only after review.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    ideal_answer: str
    ground_truth_sources: list[SourceRef]
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    source_type: GoldenSourceType
    ground_truth_parent_ids: list[str] = Field(default_factory=list)
    ground_truth_child_chunk_ids: list[str] = Field(default_factory=list)
    human_reviewed: bool = False


# --- Retrieval ---------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """One scored, ranked chunk returned by retrieval."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_id: str | None = None
    score: float
    rank: int


class RetrievalResult(BaseModel):
    """Ranked retrieval output plus the parent IDs expanded for generation context."""

    model_config = ConfigDict(extra="forbid")

    query: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    context_parent_ids: list[str] = Field(default_factory=list)


# --- Eval report (single canonical shape consumed by evals/rag_eval.py) ------


class VariantConfig(BaseModel):
    """The explicit retrieval knobs for one evaluated variant (D3.8, D3.9, D3.10)."""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str
    chunking: str  # "parent_child" | "naive_fixed"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 30
    rerank_top_k: int | None = None
    final_context_chunks: int = 5
    dense_weight: float | None = None
    sparse_weight: float | None = None
    multi_query: int = 0
    use_hybrid: bool = False
    use_rerank: bool = False


class RetrievalMetrics(BaseModel):
    """Retrieval quality on the golden set (D3.3 selection metrics)."""

    model_config = ConfigDict(extra="forbid")

    hit_at_5: float
    mrr_at_10: float
    num_examples: int


class GenerationMetrics(BaseModel):
    """Aggregated judge scores over the golden set (rubric dims + pass rate)."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float
    answer_relevancy: float
    context_usefulness: float
    completeness: float
    refusal_quality: float
    overall: float
    pass_rate: float
    num_examples: int


class VariantReport(BaseModel):
    """Metrics for one named variant/config."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: VariantConfig
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None = None


class RagEvalReport(BaseModel):
    """Canonical RAG eval report written to artifacts/evals/rag_eval_report.json."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    variants: list[VariantReport] = Field(default_factory=list)
    embedding_comparison: list[VariantReport] = Field(default_factory=list)
    weight_sweep: list[VariantReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
