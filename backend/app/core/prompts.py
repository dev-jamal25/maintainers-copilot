"""Version-controlled prompt loader (CLAUDE.md: prompts live in files, not inline strings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Read a prompt file from ``backend/prompts`` by file name (e.g. 'rag_answer_v1.md')."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")
