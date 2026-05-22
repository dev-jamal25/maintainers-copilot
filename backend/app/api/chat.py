"""Chat API: a non-streaming turn (POST /chat) and an SSE stream (POST /chat/stream).

Routes only coordinate HTTP: resolve the user, ensure the conversation, persist the user message,
run the single tool-calling loop (ChatService), persist the assistant reply, and (for /chat/stream)
emit named SSE events. All orchestration lives in services; the heavy components are built in DI
providers that tests override, so the HTTP/SSE + auth layer is verifiable without live backends.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.auth import CurrentUserDep
from app.api.deps import (
    ModelServerDep,
    RequestIdDep,
    SessionDep,
    TracingDep,
    get_settings,
)
from app.core.config import Settings
from app.domain.chat import ChatRequest, SSEEventType
from app.domain.tools import ToolName
from app.infra.llm import AnthropicLLMClient
from app.infra.model_server_client import ModelServerClient
from app.infra.redis_cache import ConversationCache, build_redis_client
from app.infra.secrets import load_anthropic_api_key
from app.infra.tracing import TracingClient
from app.repositories.audit_repository import AuditRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episodic_memory_repository import EpisodicMemoryRepository
from app.services.chat_service import ChatService
from app.services.chat_store import ChatStore
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService
from app.services.tool_service import ToolService

router = APIRouter(tags=["chat"])


def build_chat_service(
    session: AsyncSession,
    settings: Settings,
    model_server: ModelServerClient,
    *,
    allowed_tools: frozenset[ToolName] | None = None,
    tracing: TracingClient | None = None,
) -> ChatService:
    """Assemble the tool-calling loop from shared adapters + per-request repositories.

    ``allowed_tools`` restricts which tools the model can call (None = all). The widget's anonymous
    chat passes the widget's enabled-tools subset (minus write_memory).
    """
    llm = AnthropicLLMClient(
        api_key=load_anthropic_api_key(settings.vault),
        model=settings.llm.model,
        timeout=settings.llm.timeout_seconds,
    )
    rag = RagService(chunks=ChunkRepository(session), model_server=model_server, llm=llm)
    memory = MemoryService(
        memories=EpisodicMemoryRepository(session),
        embedder=model_server,
        audit=AuditRepository(session),
    )
    tools = ToolService(model_server=model_server, rag=rag, memory=memory)
    return ChatService(
        llm=llm,
        tools=tools,
        max_tokens=settings.llm.max_tokens,
        allowed_tools=allowed_tools,
        tracing=tracing,
    )


class ChatEventOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    data: dict[str, Any]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    answer: str
    events: list[ChatEventOut]


def get_chat_service(
    session: SessionDep,
    settings: Annotated[Settings, Depends(get_settings)],
    model_server: ModelServerDep,
    tracing: TracingDep,
) -> ChatService:
    """Authenticated chat: the full tool set, shared model-server client. Overridden in tests."""
    return build_chat_service(session, settings, model_server, tracing=tracing)


# A factory so callers (the widget's anonymous chat) can build a tool-restricted service.
ChatServiceFactory = Callable[[frozenset[ToolName] | None], ChatService]


def get_chat_service_factory(
    session: SessionDep,
    settings: Annotated[Settings, Depends(get_settings)],
    model_server: ModelServerDep,
    tracing: TracingDep,
) -> ChatServiceFactory:
    def factory(allowed_tools: frozenset[ToolName] | None) -> ChatService:
        return build_chat_service(
            session, settings, model_server, allowed_tools=allowed_tools, tracing=tracing
        )

    return factory


def get_chat_store(
    session: SessionDep, settings: Annotated[Settings, Depends(get_settings)]
) -> ChatStore:
    cache = ConversationCache(
        build_redis_client(settings.redis.url),
        ttl_seconds=settings.redis.short_term_ttl_seconds,
    )
    return ChatStore(session, cache=cache)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ChatServiceFactoryDep = Annotated[ChatServiceFactory, Depends(get_chat_service_factory)]
ChatStoreDep = Annotated[ChatStore, Depends(get_chat_store)]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: CurrentUserDep,
    request_id: RequestIdDep,
    chat_service: ChatServiceDep,
    store: ChatStoreDep,
) -> ChatResponse:
    conversation_id = await store.ensure_conversation(
        payload.conversation_id, user_id=user.id, first_message=payload.message
    )
    history = await store.load_history(conversation_id, user_id=user.id)
    await store.add_user_message(conversation_id, payload.message, user_id=user.id)
    result = await chat_service.run_turn(
        user_id=user.id,
        history=history,
        user_message=payload.message,
        request_id=request_id,
        trace_id=request_id,
    )
    await store.add_assistant_message(conversation_id, result.answer, user_id=user.id)
    await store.commit()
    return ChatResponse(
        conversation_id=conversation_id,
        answer=result.answer,
        events=[ChatEventOut(type=event.type.value, data=event.data) for event in result.events],
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    user: CurrentUserDep,
    request_id: RequestIdDep,
    chat_service: ChatServiceDep,
    store: ChatStoreDep,
) -> EventSourceResponse:
    conversation_id = await store.ensure_conversation(
        payload.conversation_id, user_id=user.id, first_message=payload.message
    )
    history = await store.load_history(conversation_id, user_id=user.id)
    await store.add_user_message(conversation_id, payload.message, user_id=user.id)

    async def event_generator() -> Any:
        final_answer = ""
        yield {
            "event": "conversation",
            "data": json.dumps({"conversation_id": str(conversation_id)}),
        }
        async for event in chat_service.stream(
            user_id=user.id,
            history=history,
            user_message=payload.message,
            request_id=request_id,
            trace_id=request_id,
        ):
            if event.type in (SSEEventType.TOKEN, SSEEventType.DONE):
                final_answer = str(event.data.get("text", final_answer))
            yield {"event": event.type.value, "data": json.dumps(event.data)}
        await store.add_assistant_message(conversation_id, final_answer, user_id=user.id)
        await store.commit()

    return EventSourceResponse(event_generator())
