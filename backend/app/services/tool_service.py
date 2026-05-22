"""The chatbot's tool layer: validate typed input, call a service/adapter, return typed output.

CLAUDE.md: a single tool-calling LLM, five tools, each with typed I/O validated at the boundary;
implementations call services/adapters (never route logic); a failure returns a structured
``ToolError`` (so the chatbot never crashes — the LLM explains or falls back). ``tool_specs()``
exposes the Anthropic tool definitions derived from the domain input schemas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError

from app.domain.exceptions import AppError
from app.domain.memory import EpisodicMemory, MemoryWriteRequest
from app.domain.tools import (
    AnswerWithRagInput,
    AnswerWithRagOutput,
    ClassifyIssueInput,
    ClassifyIssueOutput,
    Entity,
    ExtractEntitiesInput,
    ExtractEntitiesOutput,
    IssueClass,
    SummarizeThreadInput,
    SummarizeThreadOutput,
    ToolError,
    ToolName,
    WriteMemoryInput,
    WriteMemoryOutput,
)
from app.infra.errors import ModelServerError

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.CLASSIFY_ISSUE: "Classify a GitHub issue as bug, feature, docs, or question.",
    ToolName.EXTRACT_ENTITIES: "Extract code-shaped named entities from issue text.",
    ToolName.SUMMARIZE_THREAD: "Summarize a long issue thread into a concise summary.",
    ToolName.ANSWER_WITH_RAG: "Answer a question grounded in project docs and resolved issues.",
    ToolName.WRITE_MEMORY: "Persist an explicit long-term memory about the user or a decision.",
}
_TOOL_INPUTS: dict[ToolName, type[BaseModel]] = {
    ToolName.CLASSIFY_ISSUE: ClassifyIssueInput,
    ToolName.EXTRACT_ENTITIES: ExtractEntitiesInput,
    ToolName.SUMMARIZE_THREAD: SummarizeThreadInput,
    ToolName.ANSWER_WITH_RAG: AnswerWithRagInput,
    ToolName.WRITE_MEMORY: WriteMemoryInput,
}


def tool_specs() -> list[dict[str, Any]]:
    """Anthropic tool definitions (name/description/input_schema) for every tool."""
    return [
        {
            "name": tool.value,
            "description": _TOOL_DESCRIPTIONS[tool],
            "input_schema": _TOOL_INPUTS[tool].model_json_schema(),
        }
        for tool in ToolName
    ]


@dataclass(frozen=True)
class ToolOutcome:
    result: BaseModel  # a typed *Output model on success, or a ToolError on failure
    is_error: bool


class ModelServerToolPort(Protocol):
    async def classify(self, title: str, issue_body: str = ...) -> tuple[str, float | None]: ...
    async def extract_entities(
        self, text: str, *, max_entities: int = ...
    ) -> list[dict[str, str]]: ...
    async def summarize(self, text: str) -> str: ...


class RagPort(Protocol):
    async def answer(
        self, payload: AnswerWithRagInput, *, request_id: str | None = ...
    ) -> AnswerWithRagOutput: ...


class MemoryPort(Protocol):
    async def write_memory(
        self,
        user_id: UUID,
        request: MemoryWriteRequest,
        *,
        request_id: str | None = ...,
        trace_id: str | None = ...,
    ) -> EpisodicMemory: ...


class ToolService:
    def __init__(
        self, *, model_server: ModelServerToolPort, rag: RagPort, memory: MemoryPort
    ) -> None:
        self._model_server = model_server
        self._rag = rag
        self._memory = memory

    async def run(
        self,
        tool: ToolName,
        raw_input: dict[str, Any],
        *,
        user_id: UUID,
        request_id: str | None = None,
    ) -> ToolOutcome:
        input_model = _TOOL_INPUTS[tool]
        try:
            parsed = input_model.model_validate(raw_input)
        except ValidationError as exc:
            return self._error(tool, "invalid_input", str(exc), recoverable=False)

        try:
            return await self._dispatch(tool, parsed, user_id=user_id, request_id=request_id)
        except ModelServerError as exc:
            return self._error(tool, "model_server_unavailable", str(exc), recoverable=True)
        except AppError as exc:
            return self._error(tool, exc.code, exc.message, recoverable=True)

    async def _dispatch(
        self,
        tool: ToolName,
        parsed: BaseModel,
        *,
        user_id: UUID,
        request_id: str | None,
    ) -> ToolOutcome:
        if isinstance(parsed, ClassifyIssueInput):
            label, confidence = await self._model_server.classify(parsed.title, parsed.body)
            try:
                issue_class = IssueClass(label)
            except ValueError:
                return self._error(
                    tool,
                    "unexpected_label",
                    f"model returned unknown label '{label}'",
                    recoverable=False,
                )
            return ToolOutcome(
                ClassifyIssueOutput(label=issue_class, confidence=confidence), is_error=False
            )

        if isinstance(parsed, ExtractEntitiesInput):
            raw = await self._model_server.extract_entities(
                parsed.text, max_entities=parsed.max_entities
            )
            entities = [Entity(text=e["text"], type=e["type"]) for e in raw]
            return ToolOutcome(ExtractEntitiesOutput(entities=entities), is_error=False)

        if isinstance(parsed, SummarizeThreadInput):
            summary = await self._model_server.summarize(parsed.text)
            return ToolOutcome(SummarizeThreadOutput(summary=summary), is_error=False)

        if isinstance(parsed, AnswerWithRagInput):
            output = await self._rag.answer(parsed, request_id=request_id)
            return ToolOutcome(output, is_error=False)

        if isinstance(parsed, WriteMemoryInput):
            memory = await self._memory.write_memory(
                user_id,
                MemoryWriteRequest(
                    event_type=parsed.event_type, content=parsed.content, subject=parsed.subject
                ),
                request_id=request_id,
            )
            return ToolOutcome(
                WriteMemoryOutput(memory_id=str(memory.id), stored=True), is_error=False
            )

        return self._error(tool, "unknown_tool", f"no handler for {tool}", recoverable=False)

    def _error(self, tool: ToolName, code: str, message: str, *, recoverable: bool) -> ToolOutcome:
        logger.warning("tool_error", extra={"tool": tool.value, "code": code})
        return ToolOutcome(
            ToolError(tool=tool, code=code, message=message, recoverable=recoverable),
            is_error=True,
        )
