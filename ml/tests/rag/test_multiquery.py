from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.multiquery import (  # noqa: E402
    DEFAULT_MULTI_QUERY,
    build_prompt,
    expand_queries,
    generate_rewrites,
    load_multiquery_prompt,
    parse_rewrites,
)


def test_default_multi_query_is_three() -> None:
    assert DEFAULT_MULTI_QUERY == 3


def test_parse_rewrites_strips_numbering_bullets_quotes_and_caps() -> None:
    raw = '1. how to fix scheduler\n2) scheduler not running\n- restart scheduler\n"extra query"'
    assert parse_rewrites(raw, 3) == [
        "how to fix scheduler",
        "scheduler not running",
        "restart scheduler",
    ]


def test_parse_rewrites_deduplicates() -> None:
    assert parse_rewrites("alpha\nalpha\nbeta\ngamma", 3) == ["alpha", "beta", "gamma"]


def test_expand_queries_keeps_original_first_and_dedupes() -> None:
    assert expand_queries("q", ["q", "a", "b", "a"]) == ["q", "a", "b"]


def test_generate_rewrites_uses_injected_function() -> None:
    def fake_rewrite_fn(query: str, n: int) -> str:
        return "scheduler stuck\nscheduler not scheduling\nrestart scheduler service"

    assert generate_rewrites("scheduler problem", fake_rewrite_fn, n=3) == [
        "scheduler stuck",
        "scheduler not scheduling",
        "restart scheduler service",
    ]


def test_build_prompt_substitutes_placeholders() -> None:
    assert build_prompt("Give {n} queries for: {question}", "fix dag", 3) == (
        "Give 3 queries for: fix dag"
    )


def test_multiquery_prompt_file_has_placeholders() -> None:
    template = load_multiquery_prompt()
    assert "{n}" in template
    assert "{question}" in template
