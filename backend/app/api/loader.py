"""Serves the widget loader script at ``/widget.js`` (one <script> tag embeds the widget)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Response

router = APIRouter(tags=["widget"])

_LOADER_PATH = Path(__file__).resolve().parents[1] / "static" / "widget_loader.js"


@lru_cache(maxsize=1)
def _loader_js() -> str:
    return _LOADER_PATH.read_text(encoding="utf-8")


@router.get("/widget.js")
async def widget_loader() -> Response:
    return Response(
        content=_loader_js(),
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=300"},
    )
