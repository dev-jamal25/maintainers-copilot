"""SQL-only repository for ``widgets`` (embeddable widget configs keyed by public widget_id)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.widget import WidgetConfig, WidgetCreate, WidgetPosition, WidgetTheme

_INSERT_SQL = text(
    """
    INSERT INTO widgets (
        widget_id, allowed_origins, theme, primary_color, position, greeting,
        enabled_tools, is_active, created_by, created_at, updated_at
    ) VALUES (
        :widget_id, CAST(:allowed_origins AS jsonb), :theme, :primary_color, :position, :greeting,
        CAST(:enabled_tools AS jsonb), :is_active, :created_by, :created_at, :created_at
    )
    """
)

_GET_SQL = text(
    """
    SELECT widget_id, allowed_origins, theme, primary_color, position, greeting,
           enabled_tools, is_active, created_by, created_at, updated_at
    FROM widgets WHERE widget_id = :widget_id
    """
)
_LIST_SQL = text(
    """
    SELECT widget_id, allowed_origins, theme, primary_color, position, greeting,
           enabled_tools, is_active, created_by, created_at, updated_at
    FROM widgets ORDER BY created_at DESC
    """
)


def _to_config(row: dict[str, Any]) -> WidgetConfig:
    return WidgetConfig(
        widget_id=str(row["widget_id"]),
        allowed_origins=list(row["allowed_origins"] or []),
        theme=WidgetTheme(row["theme"]),
        primary_color=str(row["primary_color"]),
        position=WidgetPosition(row["position"]),
        greeting=str(row["greeting"]),
        enabled_tools=list(row["enabled_tools"] or []),
        is_active=bool(row["is_active"]),
        created_by=row["created_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class WidgetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, widget_id: str, payload: WidgetCreate, created_by: UUID
    ) -> WidgetConfig:
        now = datetime.now(UTC)
        await self._session.execute(
            _INSERT_SQL,
            {
                "widget_id": widget_id,
                "allowed_origins": json.dumps(payload.allowed_origins),
                "theme": payload.theme.value,
                "primary_color": payload.primary_color,
                "position": payload.position.value,
                "greeting": payload.greeting,
                "enabled_tools": json.dumps(payload.enabled_tools),
                "is_active": payload.is_active,
                "created_by": created_by,
                "created_at": now,
            },
        )
        return WidgetConfig(
            widget_id=widget_id,
            allowed_origins=list(payload.allowed_origins),
            theme=payload.theme,
            primary_color=payload.primary_color,
            position=payload.position,
            greeting=payload.greeting,
            enabled_tools=list(payload.enabled_tools),
            is_active=payload.is_active,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

    async def get(self, widget_id: str) -> WidgetConfig | None:
        result = await self._session.execute(_GET_SQL, {"widget_id": widget_id})
        row = result.mappings().first()
        return _to_config(dict(row)) if row is not None else None

    async def list_all(self) -> list[WidgetConfig]:
        result = await self._session.execute(_LIST_SQL)
        return [_to_config(dict(row)) for row in result.mappings().all()]
