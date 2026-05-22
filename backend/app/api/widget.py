"""Widget config API: public config (CORS per allowed_origins), embed page (CSP), admin CRUD.

Public routes are consumed by the loader/widget in the browser; admin routes are gated by
``require_admin``. CORS + CSP are derived from each widget's DB ``allowed_origins`` — never env.
"""

from __future__ import annotations

import json
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from app.api.auth import AdminDep
from app.api.chat import ChatEventOut, ChatServiceFactoryDep
from app.api.deps import RequestIdDep, SessionDep, SettingsDep
from app.domain.chat import SSEEventType
from app.domain.exceptions import PermissionDenied
from app.domain.tools import ToolName
from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPublicConfig
from app.infra.redis_cache import ConversationCache, build_redis_client
from app.repositories.widget_repository import WidgetRepository
from app.services.widget_service import (
    WidgetService,
    csp_frame_ancestors,
    is_origin_allowed,
)

router = APIRouter(tags=["widget"])

# Anonymous widget visitors act as the nil user and can never write long-term memory.
_NIL_USER = UUID(int=0)
_ANON_BLOCKED_TOOLS = frozenset({ToolName.WRITE_MEMORY})


def _anonymous_allowed_tools(config: WidgetConfig) -> frozenset[ToolName]:
    enabled = {tool for tool in ToolName if tool.value in set(config.enabled_tools)}
    return frozenset(enabled - _ANON_BLOCKED_TOOLS)


def get_widget_service(session: SessionDep) -> WidgetService:
    return WidgetService(WidgetRepository(session))


WidgetServiceDep = Annotated[WidgetService, Depends(get_widget_service)]


@router.get("/widgets/{widget_id}/config", response_model=WidgetPublicConfig)
async def get_widget_config(
    widget_id: str, request: Request, response: Response, service: WidgetServiceDep
) -> WidgetPublicConfig:
    config = await service.get_config(widget_id)
    origin = request.headers.get("origin")
    if is_origin_allowed(config.allowed_origins, origin) and origin is not None:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return config.to_public()


@router.get("/widgets/{widget_id}/embed", response_class=HTMLResponse)
async def embed_widget(
    widget_id: str, request: Request, service: WidgetServiceDep, settings: SettingsDep
) -> HTMLResponse:
    config = await service.get_config(widget_id)
    public = config.to_public()
    bundle = settings.widget.public_url.rstrip("/")
    api_base = str(request.base_url).rstrip("/")
    # The React bundle (served by the widget service) mounts on the root div; data-* attributes give
    # it the widget id + API base. CSP frame-ancestors limits who may iframe this page (DB origins).
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Maintainer's Copilot</title>"
        f"<link rel='stylesheet' href='{bundle}/assets/widget.css'></head><body>"
        f"<div id='maintainers-copilot-root' data-widget-id='{public.widget_id}' "
        f"data-api-base='{api_base}'>{public.greeting}</div>"
        f"<script type='module' src='{bundle}/assets/widget.js'></script>"
        "</body></html>"
    )
    return HTMLResponse(
        content=html,
        headers={"Content-Security-Policy": csp_frame_ancestors(config.allowed_origins)},
    )


@router.post("/widgets", response_model=WidgetConfig, status_code=201)
async def create_widget(
    payload: WidgetCreate, admin: AdminDep, service: WidgetServiceDep, session: SessionDep
) -> WidgetConfig:
    widget = await service.create(payload, created_by=admin.id)
    await session.commit()
    return widget


@router.get("/widgets", response_model=list[WidgetConfig])
async def list_widgets(admin: AdminDep, service: WidgetServiceDep) -> list[WidgetConfig]:
    return await service.list_widgets()


# --- anonymous widget chat (public visitors, no auth) ------------------------


def get_conversation_cache(settings: SettingsDep) -> ConversationCache:
    return ConversationCache(
        build_redis_client(settings.redis.url),
        ttl_seconds=settings.redis.short_term_ttl_seconds,
    )


ConversationCacheDep = Annotated[ConversationCache, Depends(get_conversation_cache)]


class WidgetChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)
    session_id: str | None = Field(default=None, max_length=64)


class WidgetChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    answer: str
    events: list[ChatEventOut]


async def _prepare_widget_chat(
    widget_id: str, request: Request, service: WidgetService
) -> tuple[frozenset[ToolName], str | None]:
    """Validate the widget is active and the request origin is allowed; return tools + origin."""
    config = await service.get_config(widget_id)  # 404 if missing
    if not config.is_active:
        raise PermissionDenied("This widget is not active.")
    origin = request.headers.get("origin")
    if not is_origin_allowed(config.allowed_origins, origin):
        raise PermissionDenied("Origin is not allowed for this widget.")
    return _anonymous_allowed_tools(config), origin


@router.post("/widgets/{widget_id}/chat", response_model=WidgetChatResponse)
async def widget_chat(
    widget_id: str,
    payload: WidgetChatRequest,
    request: Request,
    response: Response,
    service: WidgetServiceDep,
    cache: ConversationCacheDep,
    make_chat: ChatServiceFactoryDep,
    request_id: RequestIdDep,
) -> WidgetChatResponse:
    allowed_tools, origin = await _prepare_widget_chat(widget_id, request, service)
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    session_id = payload.session_id or uuid4().hex
    cache_user = f"widget:{widget_id}"
    history = await cache.load(cache_user, session_id)
    chat = make_chat(allowed_tools)
    result = await chat.run_turn(
        user_id=_NIL_USER,
        history=history,
        user_message=payload.message,
        request_id=request_id,
        trace_id=request_id,
    )
    await cache.append(cache_user, session_id, {"role": "user", "content": payload.message})
    await cache.append(cache_user, session_id, {"role": "assistant", "content": result.answer})
    return WidgetChatResponse(
        session_id=session_id,
        answer=result.answer,
        events=[ChatEventOut(type=event.type.value, data=event.data) for event in result.events],
    )


@router.post("/widgets/{widget_id}/chat/stream")
async def widget_chat_stream(
    widget_id: str,
    payload: WidgetChatRequest,
    request: Request,
    service: WidgetServiceDep,
    cache: ConversationCacheDep,
    make_chat: ChatServiceFactoryDep,
    request_id: RequestIdDep,
) -> EventSourceResponse:
    allowed_tools, origin = await _prepare_widget_chat(widget_id, request, service)
    session_id = payload.session_id or uuid4().hex
    cache_user = f"widget:{widget_id}"
    history = await cache.load(cache_user, session_id)
    chat = make_chat(allowed_tools)
    headers = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}

    async def event_generator() -> Any:
        final_answer = ""
        yield {"event": "session", "data": json.dumps({"session_id": session_id})}
        async for event in chat.stream(
            user_id=_NIL_USER,
            history=history,
            user_message=payload.message,
            request_id=request_id,
            trace_id=request_id,
        ):
            if event.type in (SSEEventType.TOKEN, SSEEventType.DONE):
                final_answer = str(event.data.get("text", final_answer))
            yield {"event": event.type.value, "data": json.dumps(event.data)}
        await cache.append(cache_user, session_id, {"role": "user", "content": payload.message})
        await cache.append(cache_user, session_id, {"role": "assistant", "content": final_answer})

    return EventSourceResponse(event_generator(), headers=headers)
