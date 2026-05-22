"""The single tool-calling LLM loop (CLAUDE.md: one tool-calling LLM, not a workflow/multi-agent).

One assistant turn may call tools; results are fed back and the loop continues until the model
produces a final answer or the bounded iteration limit is hit. Emits named events (``SSEEventType``)
so the route can stream them; ``run_turn`` collects them for the non-streaming path. Tool failures
are surfaced as ``tool_result`` events with the structured error (the model decides how to recover);
the loop never crashes. The service is pure (no DB/HTTP) so it unit-tests with fakes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.prompts import load_prompt
from app.domain.chat import SSEEventType
from app.domain.tools import ToolName
from app.infra.llm import ToolCallingLLM
from app.infra.redaction import redact_value
from app.infra.tracing import TracingClient, traced
from app.services.tool_service import ToolService, tool_specs

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "chat_system_v1.md"


@dataclass(frozen=True)
class ChatEvent:
    type: SSEEventType
    data: dict[str, Any]


@dataclass
class ChatResult:
    answer: str
    events: list[ChatEvent] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        *,
        llm: ToolCallingLLM,
        tools: ToolService,
        max_iterations: int = 5,
        max_tokens: int = 1024,
        allowed_tools: frozenset[ToolName] | None = None,
        tracing: TracingClient | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._allowed_tools = allowed_tools
        self._tracing = tracing

    async def stream(
        self,
        *,
        user_id: UUID,
        history: list[dict[str, Any]],
        user_message: str,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[ChatEvent]:
        system = load_prompt(_SYSTEM_PROMPT)
        specs = tool_specs(self._allowed_tools)
        messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_message}]

        for _ in range(self._max_iterations):
            async with traced(self._tracing, "chat.llm_turn", trace_id=trace_id):
                turn = await self._llm.converse(
                    system=system, messages=messages, tools=specs, max_tokens=self._max_tokens
                )
            messages.append({"role": "assistant", "content": self._assistant_content(turn)})

            if not turn.tool_calls:
                yield ChatEvent(SSEEventType.TOKEN, {"text": turn.text})
                yield ChatEvent(SSEEventType.DONE, {"text": turn.text})
                return

            tool_result_blocks: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                yield ChatEvent(
                    SSEEventType.TOOL_CALL,
                    {"tool": call.name, "input": redact_value(call.input)},
                )
                async with traced(
                    self._tracing,
                    f"chat.tool.{call.name}",
                    trace_id=trace_id,
                    metadata={"tool": call.name},
                ):
                    payload, is_error = await self._run_tool(
                        call.name, call.input, user_id=user_id, request_id=request_id
                    )
                yield ChatEvent(
                    SSEEventType.TOOL_RESULT,
                    {"tool": call.name, "is_error": is_error, "result": redact_value(payload)},
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(payload),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})

        yield ChatEvent(
            SSEEventType.ERROR,
            {"message": "Reached the tool-call limit without a final answer."},
        )

    async def run_turn(
        self,
        *,
        user_id: UUID,
        history: list[dict[str, Any]],
        user_message: str,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> ChatResult:
        result = ChatResult(answer="")
        async for event in self.stream(
            user_id=user_id,
            history=history,
            user_message=user_message,
            request_id=request_id,
            trace_id=trace_id,
        ):
            result.events.append(event)
            if event.type in (SSEEventType.TOKEN, SSEEventType.DONE):
                result.answer = str(event.data.get("text", result.answer))
        return result

    @staticmethod
    def _assistant_content(turn: object) -> list[dict[str, Any]] | str:
        from app.infra.llm import LLMTurn

        assert isinstance(turn, LLMTurn)
        blocks: list[dict[str, Any]] = []
        if turn.text:
            blocks.append({"type": "text", "text": turn.text})
        for call in turn.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
        return blocks or turn.text

    async def _run_tool(
        self, name: str, raw_input: dict[str, Any], *, user_id: UUID, request_id: str | None
    ) -> tuple[dict[str, Any], bool]:
        try:
            tool = ToolName(name)
        except ValueError:
            return {"error": {"code": "unknown_tool", "message": f"no tool named {name}"}}, True
        if self._allowed_tools is not None and tool not in self._allowed_tools:
            return {"error": {"code": "tool_not_allowed", "message": f"{name} is disabled"}}, True
        outcome = await self._tools.run(tool, raw_input, user_id=user_id, request_id=request_id)
        return outcome.result.model_dump(mode="json"), outcome.is_error
