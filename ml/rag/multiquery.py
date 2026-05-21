"""Multi-query transformation (A10, DECISIONS D3.9 / D1.10).

Generates 3 query rewrites by default (Claude Haiku 4.5) so retrieval searches the same intent
from several angles. The prompt is the version-controlled ``backend/prompts/rag_multiquery_v1.md``.

Parsing/expansion are pure and unit-tested; the Anthropic call is injected via ``rewrite_fn`` so
CI/tests never make live LLM calls (the eval supplies a cached or real rewrite function).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "backend" / "prompts" / "rag_multiquery_v1.md"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MULTI_QUERY = 3

# (query, n) -> raw model text containing one rewrite per line.
RewriteFn = Callable[[str, int], str]

_LEADING = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)?")


def load_multiquery_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, query: str, n: int) -> str:
    """Fill the prompt template's {n} and {question} placeholders (literal replace, brace-safe)."""
    return template.replace("{n}", str(n)).replace("{question}", query)


def parse_rewrites(raw: str, n: int) -> list[str]:
    """Parse model output into up to ``n`` clean, de-duplicated rewrite strings."""
    rewrites: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        candidate = _LEADING.sub("", line.strip()).strip().strip("\"'").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        rewrites.append(candidate)
        if len(rewrites) >= n:
            break
    return rewrites


def expand_queries(query: str, rewrites: Sequence[str]) -> list[str]:
    """Return the original query followed by unique rewrites (original always first)."""
    expanded = [query]
    seen = {query}
    for rewrite in rewrites:
        if rewrite and rewrite not in seen:
            expanded.append(rewrite)
            seen.add(rewrite)
    return expanded


def generate_rewrites(
    query: str,
    rewrite_fn: RewriteFn,
    *,
    n: int = DEFAULT_MULTI_QUERY,
) -> list[str]:
    """Generate up to ``n`` rewrites for a query using the injected rewrite function."""
    return parse_rewrites(rewrite_fn(query, n), n)


def anthropic_rewrite_fn(query: str, n: int) -> str:
    """Real Haiku rewrite function (not used in tests/CI; requires an Anthropic API key)."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = build_prompt(load_multiquery_prompt(), query, n)
    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)
