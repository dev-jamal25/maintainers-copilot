"""Widget configuration domain models.

A widget config is keyed by a public ``widget_id`` and drives the embeddable React widget. Security
boundary (CLAUDE.md): ``allowed_origins`` is used server-side only — for the widget-route CORS
allowlist and the embed-route CSP ``frame-ancestors`` — and is NEVER sent to the browser. The loader
fetches only :class:`WidgetPublicConfig`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WidgetPosition(StrEnum):
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    TOP_RIGHT = "top-right"
    TOP_LEFT = "top-left"


class WidgetTheme(StrEnum):
    LIGHT = "light"
    DARK = "dark"


class WidgetConfig(BaseModel):
    """Full server-side widget config (includes the security-sensitive ``allowed_origins``)."""

    model_config = ConfigDict(extra="forbid")

    widget_id: str
    allowed_origins: list[str] = Field(default_factory=list)
    theme: WidgetTheme = WidgetTheme.LIGHT
    primary_color: str = "#2563eb"
    position: WidgetPosition = WidgetPosition.BOTTOM_RIGHT
    greeting: str = "Hi! Ask me about this project's issues."
    enabled_tools: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    def to_public(self) -> WidgetPublicConfig:
        """Project to the browser-safe subset (drops allowed_origins + ownership/timestamps)."""
        return WidgetPublicConfig(
            widget_id=self.widget_id,
            theme=self.theme,
            primary_color=self.primary_color,
            position=self.position,
            greeting=self.greeting,
            enabled_tools=list(self.enabled_tools),
            is_active=self.is_active,
        )


class WidgetPublicConfig(BaseModel):
    """The only widget fields exposed to the browser/loader. No origins, no ownership."""

    model_config = ConfigDict(extra="forbid")

    widget_id: str
    theme: WidgetTheme
    primary_color: str
    position: WidgetPosition
    greeting: str
    enabled_tools: list[str] = Field(default_factory=list)
    is_active: bool


class WidgetCreate(BaseModel):
    """Admin payload to create/update a widget config."""

    model_config = ConfigDict(extra="forbid")

    allowed_origins: list[str] = Field(default_factory=list)
    theme: WidgetTheme = WidgetTheme.LIGHT
    primary_color: str = "#2563eb"
    position: WidgetPosition = WidgetPosition.BOTTOM_RIGHT
    greeting: str = "Hi! Ask me about this project's issues."
    enabled_tools: list[str] = Field(default_factory=list)
    is_active: bool = True
