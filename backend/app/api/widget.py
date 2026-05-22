"""Widget config API: public config (CORS per allowed_origins), embed page (CSP), admin CRUD.

Public routes are consumed by the loader/widget in the browser; admin routes are gated by
``require_admin``. CORS + CSP are derived from each widget's DB ``allowed_origins`` — never env.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse

from app.api.auth import AdminDep
from app.api.deps import SessionDep
from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPublicConfig
from app.repositories.widget_repository import WidgetRepository
from app.services.widget_service import (
    WidgetService,
    csp_frame_ancestors,
    is_origin_allowed,
)

router = APIRouter(tags=["widget"])


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
async def embed_widget(widget_id: str, service: WidgetServiceDep) -> HTMLResponse:
    config = await service.get_config(widget_id)
    public = config.to_public()
    # CSP frame-ancestors restricts who may iframe this page (from DB allowed_origins).
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Maintainer's Copilot</title></head><body>"
        f"<div id='maintainers-copilot-root' data-widget-id='{public.widget_id}'>"
        f"{public.greeting}</div>"
        "<!-- React widget bundle mounts here (served by the widget service) -->"
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
