"""Day 4 domain-model invariants: strict validation + the widget public/private boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.chat import ChatRequest, MessageRole
from app.domain.tools import (
    ClassifyIssueInput,
    ClassifyIssueOutput,
    IssueClass,
    ToolError,
    ToolName,
)
from app.domain.widget import WidgetConfig, WidgetPosition


def test_chat_request_rejects_empty_and_extra() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", bogus="x")  # type: ignore[call-arg]


def test_message_role_values() -> None:
    assert MessageRole.TOOL.value == "tool"
    assert {r.value for r in MessageRole} == {"user", "assistant", "system", "tool"}


def test_classify_output_confidence_bounds() -> None:
    assert ClassifyIssueOutput(label=IssueClass.BUG, confidence=0.9).confidence == 0.9
    with pytest.raises(ValidationError):
        ClassifyIssueOutput(label=IssueClass.BUG, confidence=1.5)


def test_classify_input_requires_title() -> None:
    with pytest.raises(ValidationError):
        ClassifyIssueInput(title="", body="x")


def test_tool_error_shape() -> None:
    err = ToolError(
        tool=ToolName.ANSWER_WITH_RAG,
        code="model_server_timeout",
        message="upstream timed out",
        recoverable=True,
    )
    assert err.recoverable is True
    assert err.tool == ToolName.ANSWER_WITH_RAG


def test_widget_public_config_excludes_allowed_origins() -> None:
    config = WidgetConfig(
        widget_id="w_demo",
        allowed_origins=["https://allowed.example"],
        primary_color="#123456",
        position=WidgetPosition.BOTTOM_LEFT,
        created_by=uuid4(),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    public = config.to_public()
    dumped = public.model_dump()
    assert "allowed_origins" not in dumped
    assert "created_by" not in dumped
    assert dumped["widget_id"] == "w_demo"
    assert dumped["primary_color"] == "#123456"
    assert dumped["position"] == "bottom-left"
