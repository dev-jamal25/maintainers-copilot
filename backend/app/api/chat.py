"""Chat API: a non-streaming turn (POST /chat) and an SSE stream (POST /chat/stream).

Routes only coordinate HTTP: resolve the user, ensure the conversation, persist the user message,
run the single tool-calling loop (ChatService), persist the assistant reply, and (for /chat/stream)
emit named SSE events. All orchestration lives in services; the heavy components are built in DI
providers that tests override, so the HTTP/SSE + auth layer is verifiable without live backends.
"""

from __future__ import annotations

import json
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse

from app.api.auth import CurrentUserDep
from app.api.deps import RequestIdDep, SessionDep, get_settings
from app.core.config import Settings
from app.domain.chat import ChatRequest, SSEEventType
from app.infra.llm import AnthropicLLMClient
from app.infra.model_server_client import ModelServerClient
from app.infra.secrets import load_anthropic_api_key
from app.repositories.audit_repository import AuditRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episodic_memory_repository import EpisodicMemoryRepository
from app.services.chat_service import ChatService
from app.services.chat_store import ChatStore
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService
from app.services.tool_service import ToolService

router = APIRouter(tags=["chat"])


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
    session: SessionDep, settings: Annotated[Settings, Depends(get_settings)]
) -> ChatService:
    """Build the tool-calling loop with real adapters. Overridden in tests."""
    model_server = ModelServerClient(settings.model_server)
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
    return ChatService(llm=llm, tools=tools, max_tokens=settings.llm.max_tokens)


def get_chat_store(session: SessionDep) -> ChatStore:
    return ChatStore(session)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
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
    history = await store.load_history(conversation_id)
    await store.add_user_message(conversation_id, payload.message)
    result = await chat_service.run_turn(
        user_id=user.id, history=history, user_message=payload.message, request_id=request_id
    )
    await store.add_assistant_message(conversation_id, result.answer)
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
    history = await store.load_history(conversation_id)
    await store.add_user_message(conversation_id, payload.message)

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
        ):
            if event.type in (SSEEventType.TOKEN, SSEEventType.DONE):
                final_answer = str(event.data.get("text", final_answer))
            yield {"event": event.type.value, "data": json.dumps(event.data)}
        await store.add_assistant_message(conversation_id, final_answer)
        await store.commit()

    return EventSourceResponse(event_generator())
