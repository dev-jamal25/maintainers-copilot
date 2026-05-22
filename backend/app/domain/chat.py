"""Chat domain models: conversations, messages, and the SSE event contract.

Pydantic + stdlib only (no SQLAlchemy/httpx) so these stay importable across layers. SQLAlchemy
rows live in ``app/db``; repositories map rows <-> these domain models. The chatbot is a single
tool-calling LLM (not a workflow), so a conversation is a flat ordered list of messages.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Who produced a message. ``TOOL`` carries a tool result back to the model."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single persisted turn in a conversation."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    tool_name: str | None = None  # set when role == TOOL
    created_at: datetime


class Conversation(BaseModel):
    """A user's conversation header (messages fetched separately)."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    """Inbound chat turn from a surface (Streamlit / widget). ``conversation_id`` omitted = new."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)
    conversation_id: UUID | None = None


class SSEEventType(StrEnum):
    """Named SSE events the chat stream emits (CLAUDE.md: SSE, named events)."""

    TOKEN = "token"  # noqa: S105 - SSE event name, not a secret; incremental assistant text
    TOOL_CALL = "tool_call"  # a tool is being invoked
    TOOL_RESULT = "tool_result"  # structured tool result (or tool error)
    DONE = "done"  # stream complete; carries conversation_id + message id
    ERROR = "error"  # user-safe error; never a stack trace
