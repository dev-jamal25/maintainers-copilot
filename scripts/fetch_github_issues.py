from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "apache/airflow"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "github_issues.jsonl"
DEFAULT_MAPPING_PATH = REPO_ROOT / "ml" / "classification" / "label_mapping.yaml"
GITHUB_API_ROOT = "https://api.github.com"
PER_PAGE = 100
MAX_RETRIES = 3
TARGET_CLASS_COUNT = 4
DEFAULT_TARGET_TOTAL_ISSUES = 2000
DEFAULT_TARGET_ISSUES_PER_CLASS = 500
UNAUTHENTICATED_MAX_ISSUES = 100
UNAUTHENTICATED_MAX_PAGES = 1


@dataclass(frozen=True)
class LabelFetchTarget:
    target_label: str
    github_labels: list[str]


@dataclass(frozen=True)
class FetchSummary:
    total_fetched: int
    pull_requests_skipped: int
    output_path: Path
    rate_limit_remaining: str | None
    per_class_counts: dict[str, int]
    per_class_target: int


def load_local_env() -> None:
    """Load repo-root .env explicitly before reading dataset env vars."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def is_pull_request_item(item: JsonObject) -> bool:
    return "pull_request" in item


def load_label_fetch_targets(
    path: Path = DEFAULT_MAPPING_PATH,
) -> tuple[str, list[LabelFetchTarget]]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Label mapping must be a mapping: {path}")

    repo = raw.get("repo")
    github_label_mapping = raw.get("github_label_mapping")
    if not isinstance(repo, str):
        raise ValueError("label_mapping.yaml must define repo")
    if not isinstance(github_label_mapping, dict):
        raise ValueError("label_mapping.yaml must define github_label_mapping")

    targets: list[LabelFetchTarget] = []
    for target_label, github_labels in github_label_mapping.items():
        if not isinstance(target_label, str):
            raise ValueError("github_label_mapping keys must be strings")
        if not isinstance(github_labels, list) or not all(
            isinstance(label, str) for label in github_labels
        ):
            raise ValueError(f"github_label_mapping.{target_label} must be a list of strings")
        targets.append(LabelFetchTarget(target_label=target_label, github_labels=github_labels))

    if len(targets) != TARGET_CLASS_COUNT:
        raise ValueError(f"Expected {TARGET_CLASS_COUNT} target classes in label mapping")
    return repo, targets


def extract_label_names(labels: object) -> list[str]:
    if not isinstance(labels, list):
        return []

    names: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str):
                names.append(name)
        elif isinstance(label, str):
            names.append(label)
    return names


def extract_issue_record(
    item: JsonObject,
    *,
    fetched_for_label: str,
    fetched_for_github_label: str,
) -> JsonObject:
    return {
        "github_id": item.get("id"),
        "number": item.get("number"),
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "labels": extract_label_names(item.get("labels")),
        "fetched_for_label": fetched_for_label,
        "fetched_for_github_label": fetched_for_github_label,
        "state": item.get("state"),
        "created_at": item.get("created_at"),
        "closed_at": item.get("closed_at"),
        "updated_at": item.get("updated_at"),
        "html_url": item.get("html_url"),
    }


def validate_fetch_policy(
    *,
    token: str | None,
    max_pages: int | None,
    max_issues: int | None,
) -> None:
    if token:
        return

    is_small_fetch = (
        max_issues is not None
        and max_issues <= UNAUTHENTICATED_MAX_ISSUES
        or max_pages is not None
        and max_pages <= UNAUTHENTICATED_MAX_PAGES
    )
    if is_small_fetch:
        return

    raise SystemExit(
        "GITHUB_TOKEN is required for larger GitHub API pulls. "
        "Add it to local .env as GITHUB_TOKEN=... or use a tiny smoke run "
        f"(--max-issues {UNAUTHENTICATED_MAX_ISSUES} or "
        f"--max-pages {UNAUTHENTICATED_MAX_PAGES})."
    )


def build_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "maintainers-copilot-dataset-fetch",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_page(
    client: httpx.AsyncClient,
    *,
    repo: str,
    label: str,
    page: int,
    headers: dict[str, str],
) -> httpx.Response:
    url = f"{GITHUB_API_ROOT}/repos/{repo}/issues"
    params = {
        "state": "closed",
        "per_page": PER_PAGE,
        "page": page,
        "labels": label,
        "sort": "created",
        "direction": "desc",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            if attempt == MAX_RETRIES:
                response.raise_for_status()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == MAX_RETRIES:
                raise

        await asyncio.sleep(min(2**attempt, 8))

    raise RuntimeError("unreachable GitHub fetch retry state")


async def fetch_closed_issues(
    *,
    repo: str,
    label_targets: list[LabelFetchTarget],
    token: str | None,
    max_pages: int | None,
    max_issues: int | None,
) -> tuple[list[JsonObject], int, str | None, dict[str, int]]:
    validate_fetch_policy(token=token, max_pages=max_pages, max_issues=max_issues)

    issues: list[JsonObject] = []
    seen_issue_ids: set[object] = set()
    pull_requests_skipped = 0
    rate_limit_remaining: str | None = None
    per_class_counts: dict[str, int] = {target.target_label: 0 for target in label_targets}
    per_class_target = resolve_per_class_target(max_issues, len(label_targets))
    headers = build_headers(token)

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for target in label_targets:
            for github_label in target.github_labels:
                page = 1
                while per_class_counts[target.target_label] < per_class_target and (
                    max_pages is None or page <= max_pages
                ):
                    response = await fetch_page(
                        client,
                        repo=repo,
                        label=github_label,
                        page=page,
                        headers=headers,
                    )
                    rate_limit_remaining = response.headers.get("x-ratelimit-remaining")
                    page_items = response.json()
                    if not isinstance(page_items, list) or not page_items:
                        break

                    for item in page_items:
                        if not isinstance(item, dict):
                            continue
                        if is_pull_request_item(item):
                            pull_requests_skipped += 1
                            continue

                        issue_id = item.get("id")
                        if issue_id in seen_issue_ids:
                            continue

                        seen_issue_ids.add(issue_id)
                        issues.append(
                            extract_issue_record(
                                item,
                                fetched_for_label=target.target_label,
                                fetched_for_github_label=github_label,
                            )
                        )
                        per_class_counts[target.target_label] += 1
                        if per_class_counts[target.target_label] >= per_class_target:
                            break

                    if len(page_items) < PER_PAGE:
                        break
                    page += 1

    return issues, pull_requests_skipped, rate_limit_remaining, per_class_counts


def resolve_per_class_target(max_issues: int | None, class_count: int) -> int:
    if max_issues is None:
        return DEFAULT_TARGET_ISSUES_PER_CLASS
    return min(DEFAULT_TARGET_ISSUES_PER_CLASS, max(1, max_issues // class_count))


def write_jsonl(records: list[JsonObject], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch closed GitHub issues to raw JSONL.")
    parser.add_argument("--max-pages", type=positive_int, default=None)
    parser.add_argument("--max-issues", type=positive_int, default=DEFAULT_TARGET_TOTAL_ISSUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    return parser.parse_args()


async def run() -> FetchSummary:
    load_local_env()
    args = parse_args()
    mapping_path = (
        args.mapping_path if args.mapping_path.is_absolute() else REPO_ROOT / args.mapping_path
    )
    mapping_repo, label_targets = load_label_fetch_targets(mapping_path)
    repo = os.getenv("GITHUB_REPO", mapping_repo).strip() or mapping_repo
    if repo == DEFAULT_REPO and mapping_repo != DEFAULT_REPO:
        repo = mapping_repo
    token = os.getenv("GITHUB_TOKEN") or None

    (
        issues,
        pull_requests_skipped,
        rate_limit_remaining,
        per_class_counts,
    ) = await fetch_closed_issues(
        repo=repo,
        label_targets=label_targets,
        token=token,
        max_pages=args.max_pages,
        max_issues=args.max_issues,
    )
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_jsonl(issues, output_path)

    return FetchSummary(
        total_fetched=len(issues),
        pull_requests_skipped=pull_requests_skipped,
        output_path=output_path,
        rate_limit_remaining=rate_limit_remaining,
        per_class_counts=per_class_counts,
        per_class_target=resolve_per_class_target(args.max_issues, len(label_targets)),
    )


def main() -> int:
    summary = asyncio.run(run())
    print(f"total fetched: {summary.total_fetched}")
    print(f"pull requests skipped: {summary.pull_requests_skipped}")
    print(f"per-class fetched counts: {summary.per_class_counts}")
    print(f"per-class target: {summary.per_class_target}")
    shortfalls = {
        label: summary.per_class_target - count
        for label, count in summary.per_class_counts.items()
        if count < summary.per_class_target
    }
    if shortfalls:
        print(f"WARNING: class fetch shortfalls: {shortfalls}")
    print(f"output path: {summary.output_path}")
    print(f"remaining GitHub rate limit: {summary.rate_limit_remaining or 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
