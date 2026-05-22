"""ChatService: the single tool-calling loop — runs a tool, feeds the result back, then answers."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.domain.chat import SSEEventType
from app.domain.tools import AnswerWithRagInput, AnswerWithRagOutput
from app.infra.llm import LLMToolCall, LLMTurn
from app.services.chat_service import ChatService
from app.services.tool_service import ToolService


class ScriptedLLM:
    """Returns pre-scripted turns in order and records the messages it was given each call."""

    def __init__(self, turns: list[LLMTurn]) -> None:
        self._turns = turns
        self.index = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMTurn:
        self.seen_messages.append([dict(m) for m in messages])
        turn = self._turns[min(self.index, len(self._turns) - 1)]
        self.index += 1
        return turn


class FakeModelServer:
    async def classify(self, title: str, issue_body: str = "") -> tuple[str, float | None]:
        return "bug", 0.9

    async def extract_entities(self, text: str, *, max_entities: int = 50) -> list[dict[str, str]]:
        return []

    async def summarize(self, text: str) -> str:
        return "summary"


class FakeRag:
    async def answer(
        self, payload: AnswerWithRagInput, *, request_id: str | None = None
    ) -> AnswerWithRagOutput:
        return AnswerWithRagOutput(answer="grounded", citations=[])


class FakeMemory:
    async def write_memory(self, user_id: Any, request: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise AssertionError("write_memory should not be called in this test")


def _chat_service(turns: list[LLMTurn]) -> tuple[ChatService, ScriptedLLM]:
    llm = ScriptedLLM(turns)
    tools = ToolService(model_server=FakeModelServer(), rag=FakeRag(), memory=FakeMemory())
    return ChatService(llm=llm, tools=tools), llm


async def test_loop_runs_tool_then_returns_final_answer() -> None:
    turns = [
        LLMTurn(
            text="",
            tool_calls=[LLMToolCall(id="t1", name="answer_with_rag", input={"question": "why?"})],
        ),
        LLMTurn(text="Final grounded answer.", tool_calls=[]),
    ]
    service, llm = _chat_service(turns)
    result = await service.run_turn(user_id=uuid4(), history=[], user_message="why does it fail?")

    assert result.answer == "Final grounded answer."
    types = [e.type for e in result.events]
    assert SSEEventType.TOOL_CALL in types
    assert SSEEventType.TOOL_RESULT in types
    assert types[-1] == SSEEventType.DONE
    # The second LLM call must have received the tool_result fed back in.
    second_call = llm.seen_messages[1]
    assert any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
        for m in second_call
    )


async def test_tool_error_is_surfaced_not_raised() -> None:
    # answer_with_rag with invalid input -> ToolError -> tool_result(is_error=True); loop continues.
    turns = [
        LLMTurn(text="", tool_calls=[LLMToolCall(id="t1", name="answer_with_rag", input={})]),
        LLMTurn(text="Sorry, I could not retrieve that.", tool_calls=[]),
    ]
    service, _ = _chat_service(turns)
    result = await service.run_turn(user_id=uuid4(), history=[], user_message="q")
    tool_results = [e for e in result.events if e.type == SSEEventType.TOOL_RESULT]
    assert tool_results and tool_results[0].data["is_error"] is True
    assert result.answer == "Sorry, I could not retrieve that."


async def test_unknown_tool_name_surfaces_error() -> None:
    turns = [
        LLMTurn(text="", tool_calls=[LLMToolCall(id="t1", name="not_a_tool", input={})]),
        LLMTurn(text="done", tool_calls=[]),
    ]
    service, _ = _chat_service(turns)
    result = await service.run_turn(user_id=uuid4(), history=[], user_message="q")
    tool_results = [e for e in result.events if e.type == SSEEventType.TOOL_RESULT]
    assert tool_results[0].data["is_error"] is True


async def test_tool_call_limit_emits_error_event() -> None:
    # LLM always asks for a tool -> never finalizes -> ERROR after max_iterations.
    looping = [
        LLMTurn(
            text="",
            tool_calls=[LLMToolCall(id="t", name="answer_with_rag", input={"question": "q"})],
        )
    ]
    llm = ScriptedLLM(looping)
    tools = ToolService(model_server=FakeModelServer(), rag=FakeRag(), memory=FakeMemory())
    service = ChatService(llm=llm, tools=tools, max_iterations=3)
    result = await service.run_turn(user_id=uuid4(), history=[], user_message="q")
    assert result.events[-1].type == SSEEventType.ERROR
