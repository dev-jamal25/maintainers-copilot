"""Memory domain models.

Two stores (CLAUDE.md):
- Short-term = active conversation state in Redis (TTL'd); represented as ``ChatMessage`` lists, so
  no separate model is needed here.
- Long-term = *episodic* memory in Postgres+pgvector: concrete events ("asked about issue X",
  "prefers style Y", "accepted summary Z"). Written ONLY via the explicit ``write_memory`` tool,
  always through redaction + an audit row. These models are that contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EpisodicEventType(StrEnum):
    """Concrete event kinds for episodic memory (kept small + reviewable)."""

    ASKED_ABOUT_ISSUE = "asked_about_issue"
    PREFERENCE = "preference"
    ACCEPTED_REPLY = "accepted_reply"
    REJECTED_REPLY = "rejected_reply"
    SAVED_SUMMARY = "saved_summary"
    FACT = "fact"


class MemoryWriteRequest(BaseModel):
    """Input to the explicit ``write_memory`` tool. ``content`` is redacted before persistence."""

    model_config = ConfigDict(extra="forbid")

    event_type: EpisodicEventType
    content: str = Field(min_length=1, max_length=4_000)
    subject: str | None = Field(
        default=None,
        max_length=200,
        description="Optional handle for the event subject, e.g. 'issue #1234' or 'style'.",
    )


class EpisodicMemory(BaseModel):
    """A stored episodic memory row (embedding lives in the DB, not exposed here)."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: UUID
    event_type: EpisodicEventType
    content: str
    subject: str | None = None
    created_at: datetime


class RetrievedMemory(BaseModel):
    """An episodic memory returned by similarity recall, with its score."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    event_type: EpisodicEventType
    content: str
    subject: str | None = None
    score: float
    created_at: datetime
