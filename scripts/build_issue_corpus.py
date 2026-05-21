"""Build the held-out issue RAG corpus from resolved Airflow issues (A04).

Reads the 100 held-out issue ids from ``rag_holdout.jsonl`` and joins them against the
structured ``github_issues_with_comments.jsonl`` (which carries each issue's ``body`` and a
``comments`` array with per-comment ``github_id`` + ``author_association``). For every held-out
issue it emits CorpusRecord rows:

- one record for the issue body (the question/context), ``github_comment_id`` = the body sentinel;
- one record per *maintainer* comment (MEMBER/OWNER/CONTRIBUTOR/COLLABORATOR), carrying the real
  comment ``github_id`` so chunk IDs are stable and answers can be grounded in maintainer text.

Leakage safety: only the ids listed in ``rag_holdout.jsonl`` are ingested, and those ids were
carved out of classifier training during Day 1 (D1.16). Output:
``data/processed/rag_issue_corpus.jsonl``.

Run:
    uv run --project backend python scripts/build_issue_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import (  # noqa: E402
    ISSUE_BODY_COMMENT_ID,
    ChunkSourceType,
    CorpusRecord,
)

JsonObject = dict[str, Any]

DEFAULT_HOLDOUT_PATH = REPO_ROOT / "data" / "splits" / "rag_holdout.jsonl"
DEFAULT_WITH_COMMENTS_PATH = (
    REPO_ROOT / "data" / "raw" / "github_issues_with_comments.jsonl"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "rag_issue_corpus.jsonl"

MAINTAINER_ASSOCIATIONS = frozenset({"MEMBER", "OWNER", "CONTRIBUTOR", "COLLABORATOR"})
MIN_COMMENT_CHARS = 20


def is_maintainer(author_association: str | None) -> bool:
    """True for maintainer-side comment authors (D1.16 grounding policy)."""
    return (author_association or "").upper() in MAINTAINER_ASSOCIATIONS


def load_holdout_ids(path: Path) -> set[int]:
    """Load the held-out issue github_ids; these are the only issues we ingest (leakage-safe)."""
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            github_id = record.get("github_id")
            if github_id is not None:
                ids.add(int(github_id))
    return ids


def _issue_label(issue: JsonObject) -> str | None:
    label = issue.get("fetched_for_label") or issue.get("label")
    return str(label) if label else None


def issue_body_record(issue: JsonObject) -> CorpusRecord:
    """Build the CorpusRecord for an issue body (title + body)."""
    github_id = int(issue["github_id"])
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    text = f"{title}\n\n{body}".strip() if title else body
    label = _issue_label(issue)
    return CorpusRecord(
        source_type=ChunkSourceType.ISSUE,
        source_id=f"issue:{github_id}",
        text=text,
        title=title or None,
        url=issue.get("html_url"),
        airflow_area=None,
        tags=[label] if label else [],
        github_issue_id=github_id,
        github_comment_id=ISSUE_BODY_COMMENT_ID,
        created_at=issue.get("created_at"),
        closed_at=issue.get("closed_at"),
    )


def comment_records(issue: JsonObject) -> list[CorpusRecord]:
    """Build CorpusRecords for substantive maintainer comments on an issue."""
    github_id = int(issue["github_id"])
    title = (issue.get("title") or "").strip() or None
    label = _issue_label(issue)
    records: list[CorpusRecord] = []
    for comment in issue.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        if not is_maintainer(comment.get("author_association")):
            continue
        body = (comment.get("body") or "").strip()
        if len(body) < MIN_COMMENT_CHARS:
            continue
        comment_id_raw = comment.get("github_id")
        if comment_id_raw is None:
            continue
        comment_id = int(comment_id_raw)
        records.append(
            CorpusRecord(
                source_type=ChunkSourceType.ISSUE,
                source_id=f"issue:{github_id}:comment:{comment_id}",
                text=body,
                title=title,
                url=comment.get("html_url") or issue.get("html_url"),
                airflow_area=None,
                tags=[label] if label else [],
                github_issue_id=github_id,
                github_comment_id=comment_id,
                created_at=comment.get("created_at"),
                closed_at=issue.get("closed_at"),
            )
        )
    return records


def build_issue_corpus_records(
    holdout_ids: set[int],
    with_comments_path: Path,
) -> tuple[list[CorpusRecord], int]:
    """Join held-out ids against the with-comments file; return (records, issues_covered)."""
    records: list[CorpusRecord] = []
    seen_issues: set[int] = set()
    with with_comments_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            issue = json.loads(line)
            github_id_raw = issue.get("github_id")
            if github_id_raw is None:
                continue
            github_id = int(github_id_raw)
            if github_id not in holdout_ids or github_id in seen_issues:
                continue
            seen_issues.add(github_id)
            records.append(issue_body_record(issue))
            records.extend(comment_records(issue))
    records.sort(key=lambda r: (r.github_issue_id or 0, r.github_comment_id or 0))
    return records, len(seen_issues)


def write_corpus_jsonl(records: list[CorpusRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = json.dumps(
                record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            handle.write(payload + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the held-out issue RAG corpus JSONL."
    )
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument(
        "--with-comments", type=Path, default=DEFAULT_WITH_COMMENTS_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    holdout_ids = load_holdout_ids(_resolve(args.holdout))
    records, covered = build_issue_corpus_records(
        holdout_ids, _resolve(args.with_comments)
    )
    write_corpus_jsonl(records, _resolve(args.output))

    body_count = sum(1 for r in records if r.github_comment_id == ISSUE_BODY_COMMENT_ID)
    comment_count = len(records) - body_count
    print(f"holdout issues: {len(holdout_ids)}")
    print(f"issues covered: {covered}")
    print(
        f"records: {len(records)} (body: {body_count}, maintainer comments: {comment_count})"
    )
    print(f"output: {_resolve(args.output)}")
    if covered < len(holdout_ids):
        missing = len(holdout_ids) - covered
        print(f"WARNING: {missing} holdout issues missing from with-comments file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
