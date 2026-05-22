"""Widget config business logic + origin/CSP security helpers.

The security boundary (CLAUDE.md): ``allowed_origins`` comes from the DB, never hardcoded env. The
widget-config route uses :func:`is_origin_allowed` to set the CORS allow-origin header per request;
the embed route uses :func:`csp_frame_ancestors` to limit who may iframe it.
The browser-facing config never includes ``allowed_origins`` (see ``WidgetConfig.to_public``).
"""

from __future__ import annotations

import secrets
from typing import Protocol
from uuid import UUID

from app.domain.exceptions import NotFoundError
from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPublicConfig


class WidgetRepoPort(Protocol):
    async def create(
        self, *, widget_id: str, payload: WidgetCreate, created_by: UUID
    ) -> WidgetConfig: ...
    async def get(self, widget_id: str) -> WidgetConfig | None: ...
    async def list_all(self) -> list[WidgetConfig]: ...


def is_origin_allowed(allowed_origins: list[str], origin: str | None) -> bool:
    """True only when a concrete request Origin is explicitly listed (no wildcards)."""
    return origin is not None and origin in allowed_origins


def csp_frame_ancestors(allowed_origins: list[str]) -> str:
    """CSP directive controlling who may frame the embed page. Empty list => deny all."""
    sources = " ".join(allowed_origins) if allowed_origins else "'none'"
    return f"frame-ancestors {sources}"


class WidgetService:
    def __init__(self, repo: WidgetRepoPort) -> None:
        self._repo = repo

    async def create(self, payload: WidgetCreate, *, created_by: UUID) -> WidgetConfig:
        widget_id = f"w_{secrets.token_hex(8)}"
        return await self._repo.create(widget_id=widget_id, payload=payload, created_by=created_by)

    async def list_widgets(self) -> list[WidgetConfig]:
        return await self._repo.list_all()

    async def get_config(self, widget_id: str) -> WidgetConfig:
        config = await self._repo.get(widget_id)
        if config is None:
            raise NotFoundError("Widget not found.")
        return config

    async def get_public_config(self, widget_id: str) -> WidgetPublicConfig:
        return (await self.get_config(widget_id)).to_public()
