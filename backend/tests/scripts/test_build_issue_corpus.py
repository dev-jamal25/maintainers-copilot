from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ISSUES = _load_script("build_issue_corpus_test_module", "scripts/build_issue_corpus.py")


def _issue(github_id: int, comments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "github_id": github_id,
        "number": github_id % 100000,
        "title": "Scheduler stuck",
        "body": "The scheduler stops emitting tasks after a while.",
        "html_url": f"https://github.com/apache/airflow/issues/{github_id}",
        "fetched_for_label": "bug",
        "label": "bug",
        "created_at": "2026-03-13T16:15:08Z",
        "closed_at": "2026-03-23T08:07:03Z",
        "comments": comments,
    }


def test_is_maintainer_associations() -> None:
    assert ISSUES.is_maintainer("MEMBER")
    assert ISSUES.is_maintainer("contributor")
    assert ISSUES.is_maintainer("COLLABORATOR")
    assert not ISSUES.is_maintainer("NONE")
    assert not ISSUES.is_maintainer(None)


def test_issue_body_record_uses_body_sentinel() -> None:
    record = ISSUES.issue_body_record(_issue(111, []))
    assert record.source_type == "issue"
    assert record.source_id == "issue:111"
    assert record.github_issue_id == 111
    assert record.github_comment_id == ISSUES.ISSUE_BODY_COMMENT_ID
    assert "Scheduler stuck" in record.text
    assert record.tags == ["bug"]


def test_comment_records_keep_only_substantive_maintainer_comments() -> None:
    issue = _issue(
        222,
        [
            {
                "github_id": 9001,
                "author_association": "MEMBER",
                "body": "Clear the stale DagRun rows, then restart the scheduler.",
                "html_url": "https://github.com/apache/airflow/issues/222#c9001",
                "created_at": "2026-03-14T00:00:00Z",
            },
            {
                "github_id": 9002,
                "author_association": "NONE",
                "body": "I have the same problem, please help me as well, thanks a lot everyone.",
                "created_at": "2026-03-15T00:00:00Z",
            },
            {
                "github_id": 9003,
                "author_association": "CONTRIBUTOR",
                "body": "short",
                "created_at": "2026-03-16T00:00:00Z",
            },
        ],
    )
    records = ISSUES.comment_records(issue)
    assert len(records) == 1
    assert records[0].source_id == "issue:222:comment:9001"
    assert records[0].github_comment_id == 9001
    assert "stale DagRun" in records[0].text


def test_build_issue_corpus_is_leakage_safe_and_sorted(tmp_path: Path) -> None:
    with_comments = tmp_path / "with_comments.jsonl"
    rows = [
        _issue(
            200,
            [
                {
                    "github_id": 5,
                    "author_association": "MEMBER",
                    "body": "Set start_date in the past so the interval has elapsed.",
                    "created_at": "2026-03-14T00:00:00Z",
                }
            ],
        ),
        # An issue NOT in the holdout set must never be ingested (leakage guard).
        _issue(
            999,
            [
                {
                    "github_id": 7,
                    "author_association": "MEMBER",
                    "body": "This is a training-pool issue that must not leak into the RAG corpus.",
                    "created_at": "2026-03-14T00:00:00Z",
                }
            ],
        ),
    ]
    with with_comments.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    records, covered = ISSUES.build_issue_corpus_records({200}, with_comments)
    issue_ids = {r.github_issue_id for r in records}
    assert issue_ids == {200}, "only held-out issue ids may be ingested"
    assert covered == 1
    # body + one maintainer comment, sorted by (issue_id, comment_id) -> body sentinel 0 first.
    assert [r.github_comment_id for r in records] == [ISSUES.ISSUE_BODY_COMMENT_ID, 5]
