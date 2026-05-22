"""Tool contracts for the single tool-calling LLM (CLAUDE.md chatbot scope).

Five tools, each with a typed Pydantic input + output validated at the boundary. Tool
implementations call services/adapters (model-server, RAG, memory) — never raw route logic. A tool
failure must NOT crash the chatbot: it returns a structured :class:`ToolError` that the LLM can
explain or fall back from. These models are pure Pydantic + stdlib (layer-portable).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.memory import EpisodicEventType


class ToolName(StrEnum):
    CLASSIFY_ISSUE = "classify_issue"
    EXTRACT_ENTITIES = "extract_entities"
    SUMMARIZE_THREAD = "summarize_thread"
    ANSWER_WITH_RAG = "answer_with_rag"
    WRITE_MEMORY = "write_memory"


class IssueClass(StrEnum):
    """The four frozen classifier labels (DECISIONS: apache/airflow triage)."""

    BUG = "bug"
    FEATURE = "feature"
    DOCS = "docs"
    QUESTION = "question"


# --- classify_issue ----------------------------------------------------------


class ClassifyIssueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=20_000)


class ClassifyIssueOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: IssueClass
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# --- extract_entities --------------------------------------------------------


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    type: str


class ExtractEntitiesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)
    max_entities: int = Field(default=50, ge=1, le=200)


class ExtractEntitiesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)


# --- summarize_thread --------------------------------------------------------


class SummarizeThreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=40_000)


class SummarizeThreadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


# --- answer_with_rag ---------------------------------------------------------


class Citation(BaseModel):
    """A retrieved-chunk reference returned alongside a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_type: str
    title: str | None = None
    url: str | None = None
    score: float


class AnswerWithRagInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4_000)
    source_type: str | None = Field(
        default=None, description="Optional metadata filter: 'docs' or 'issue'."
    )


class AnswerWithRagOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)


# --- write_memory ------------------------------------------------------------


class WriteMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EpisodicEventType
    content: str = Field(min_length=1, max_length=4_000)
    subject: str | None = Field(default=None, max_length=200)


class WriteMemoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    stored: bool = True


# --- structured tool failure -------------------------------------------------


class ToolError(BaseModel):
    """Structured tool failure handed back to the LLM instead of raising.

    ``recoverable`` tells the orchestration loop whether retrying or falling back is sensible
    (e.g. a transient model-server timeout) vs. a hard failure (invalid input).
    """

    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    code: str
    message: str
    recoverable: bool = False
