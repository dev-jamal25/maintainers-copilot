from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "apache/airflow"
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "raw" / "github_issues.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "github_issues_with_comments.jsonl"
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "processed" / "dataset_metadata.json"
GITHUB_API_ROOT = "https://api.github.com"
PER_PAGE = 100
MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class EnrichmentSummary:
    input_count: int
    output_count: int
    total_comments: int
    issues_with_comments: int
    output_path: Path
    output_sha256: str
    rate_limit_remaining: str | None


def load_local_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"{display_path(path)}:{line_number} must contain a JSON object"
                )
            records.append(parsed)
    return records


def write_jsonl(records: list[JsonObject], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def build_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "maintainers-copilot-comment-fetch",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, object] | None = None,
) -> httpx.Response:
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url, headers=headers, params=params)
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


def extract_comment_record(comment: JsonObject) -> JsonObject:
    user = comment.get("user")
    return {
        "github_id": comment.get("id"),
        "body": comment.get("body") or "",
        "author_association": comment.get("author_association"),
        "user_login": user.get("login") if isinstance(user, dict) else None,
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "html_url": comment.get("html_url"),
    }


def extract_issue_record(
    issue: JsonObject,
    source_record: JsonObject,
    comments: list[JsonObject],
) -> JsonObject:
    return {
        "github_id": issue.get("id"),
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "labels": extract_label_names(issue.get("labels")),
        "fetched_for_label": source_record.get("fetched_for_label"),
        "fetched_for_github_label": source_record.get("fetched_for_github_label"),
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "updated_at": issue.get("updated_at"),
        "html_url": issue.get("html_url"),
        "comments": comments,
    }


async def fetch_issue(
    client: httpx.AsyncClient,
    repo: str,
    number: int,
    headers: dict[str, str],
    rate_limit_remaining: list[str | None],
) -> JsonObject:
    response = await get_with_retries(
        client,
        f"{GITHUB_API_ROOT}/repos/{repo}/issues/{number}",
        headers=headers,
    )
    rate_limit_remaining[0] = response.headers.get("x-ratelimit-remaining")
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError(f"GitHub issue response for #{number} was not an object")
    return parsed


async def fetch_comments(
    client: httpx.AsyncClient,
    repo: str,
    number: int,
    headers: dict[str, str],
    rate_limit_remaining: list[str | None],
) -> list[JsonObject]:
    comments: list[JsonObject] = []
    page = 1
    while True:
        response = await get_with_retries(
            client,
            f"{GITHUB_API_ROOT}/repos/{repo}/issues/{number}/comments",
            headers=headers,
            params={"per_page": PER_PAGE, "page": page},
        )
        rate_limit_remaining[0] = response.headers.get("x-ratelimit-remaining")
        parsed = response.json()
        if not isinstance(parsed, list):
            raise ValueError(f"GitHub comments response for #{number} was not a list")
        comments.extend(
            extract_comment_record(comment)
            for comment in parsed
            if isinstance(comment, dict)
        )
        if len(parsed) < PER_PAGE:
            break
        page += 1
    return comments


async def enrich_issue(
    client: httpx.AsyncClient,
    repo: str,
    source_record: JsonObject,
    headers: dict[str, str],
    rate_limit_remaining: list[str | None],
) -> tuple[JsonObject, str | None]:
    number = source_record.get("number")
    if not isinstance(number, int):
        raise ValueError(f"record is missing integer issue number: {source_record!r}")

    issue = await fetch_issue(client, repo, number, headers, rate_limit_remaining)
    comment_count = issue.get("comments")
    comments = (
        await fetch_comments(client, repo, number, headers, rate_limit_remaining)
        if isinstance(comment_count, int) and comment_count > 0
        else []
    )
    return extract_issue_record(issue, source_record, comments), None


async def enrich_issues(
    *,
    repo: str,
    source_records: list[JsonObject],
    token: str,
    concurrency: int,
) -> tuple[list[JsonObject], str | None]:
    headers = build_headers(token)
    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_remaining: list[str | None] = [None]
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def run_one(index: int, record: JsonObject) -> tuple[int, JsonObject]:
            async with semaphore:
                enriched, _ = await enrich_issue(
                    client,
                    repo,
                    record,
                    headers,
                    rate_limit_remaining,
                )
                return index, enriched

        indexed_records = await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(source_records))
        )

    indexed_records.sort(key=lambda item: item[0])
    return [record for _, record in indexed_records], rate_limit_remaining[0]


def update_dataset_metadata(
    metadata_path: Path, output_path: Path, output_sha256: str
) -> None:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"{display_path(metadata_path)} must contain a JSON object")

    metadata["raw_input_path"] = display_path(output_path)
    metadata["raw_input_sha256"] = output_sha256
    metadata["raw_comments_enriched_at"] = (
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issue comments for the mapped raw issue pool."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--concurrency", type=positive_int, default=DEFAULT_CONCURRENCY)
    return parser.parse_args()


async def run() -> EnrichmentSummary:
    load_local_env()
    args = parse_args()
    token = os.getenv("GITHUB_TOKEN") or ""
    if not token:
        raise SystemExit(
            "GITHUB_TOKEN is required in local .env to fetch issue comments."
        )

    repo = (os.getenv("GITHUB_REPO") or DEFAULT_REPO).strip() or DEFAULT_REPO
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    metadata_path = resolve_repo_path(args.metadata_path)
    source_records = read_jsonl(input_path)

    enriched_records, rate_limit_remaining = await enrich_issues(
        repo=repo,
        source_records=source_records,
        token=token,
        concurrency=args.concurrency,
    )
    write_jsonl(enriched_records, output_path)
    output_sha256 = sha256_file(output_path)
    update_dataset_metadata(metadata_path, output_path, output_sha256)

    total_comments = sum(
        len(record.get("comments")) if isinstance(record.get("comments"), list) else 0
        for record in enriched_records
    )
    issues_with_comments = sum(
        1
        for record in enriched_records
        if isinstance(record.get("comments"), list) and len(record["comments"]) > 0
    )
    return EnrichmentSummary(
        input_count=len(source_records),
        output_count=len(enriched_records),
        total_comments=total_comments,
        issues_with_comments=issues_with_comments,
        output_path=output_path,
        output_sha256=output_sha256,
        rate_limit_remaining=rate_limit_remaining,
    )


def main() -> int:
    summary = asyncio.run(run())
    print("comment enrichment summary")
    print(f"input issues: {summary.input_count}")
    print(f"output issues: {summary.output_count}")
    print(f"issues with comments: {summary.issues_with_comments}")
    print(f"total comments: {summary.total_comments}")
    print(f"output path: {display_path(summary.output_path)}")
    print(f"output sha256: {summary.output_sha256}")
    print(f"remaining GitHub rate limit: {summary.rate_limit_remaining or 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
