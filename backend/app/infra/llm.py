"""LLM adapter (Anthropic) — the only place the app talks to the chatbot LLM.

Exposes a stable ``LLMClient`` Protocol so services depend on the surface, not the SDK, and tests
inject a fake. ``generate`` is the simple grounded-answer path (single user prompt) used by RAG;
the tool-calling chat loop (Slice 6) will add a richer method. Provider errors are wrapped as
``LLMUnavailableError`` so services convert them into structured tool errors / domain exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.infra.errors import LLMUnavailableError


class LLMClient(Protocol):
    async def generate(self, prompt: str, *, max_tokens: int) -> str:
        """Return the model's text answer for a single user prompt."""


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMTurn:
    """One assistant turn: any text, plus any tool calls it wants run (Anthropic tool use)."""

    text: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    stop_reason: str | None = None


class ToolCallingLLM(Protocol):
    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMTurn:
        """Run one assistant turn over the running message list with the available tools."""


class AnthropicLLMClient:
    """Real adapter over the async Anthropic SDK. Model + key injected (key comes from Vault)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float,
        client: object | None = None,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def generate(self, prompt: str, *, max_tokens: int) -> str:
        from anthropic import AnthropicError

        try:
            message = await self._client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicError as exc:
            raise LLMUnavailableError("Anthropic request failed") from exc

        parts: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(str(block.text))
        return "".join(parts)

    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMTurn:
        from anthropic import AnthropicError

        try:
            message = await self._client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
        except AnthropicError as exc:
            raise LLMUnavailableError("Anthropic tool-use request failed") from exc

        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(str(block.text))
            elif block_type == "tool_use":
                tool_calls.append(
                    LLMToolCall(id=str(block.id), name=str(block.name), input=dict(block.input))
                )
        return LLMTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(message, "stop_reason", None),
        )
