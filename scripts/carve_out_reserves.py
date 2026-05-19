from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPED_PATH = REPO_ROOT / "data" / "processed" / "issues_mapped.jsonl"
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "github_issues_with_comments.jsonl"
METADATA_PATH = REPO_ROOT / "data" / "processed" / "dataset_metadata.json"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
GOLDEN_PATH = SPLITS_DIR / "golden_eval.jsonl"
RAG_PATH = SPLITS_DIR / "rag_holdout.jsonl"
SPLITTABLE_PATH = SPLITS_DIR / "splittable_pool.jsonl"

RANDOM_SEED = 42
GOLDEN_COUNTS = {"bug": 7, "docs": 6, "feature": 7, "question": 5}
LABEL_ORDER = ("bug", "docs", "feature", "question")
RAG_MIN_COUNT = 50
RAG_MAX_COUNT = 100
RAG_MIN_WORDS = 80
MAINTAINER_ASSOCIATIONS = {"MEMBER", "OWNER", "CONTRIBUTOR"}
COMMENT_FIELD_TERMS = ("comment", "discussion", "thread", "reply", "replies")

GOLDEN_SELECTION_CRITERIA = (
    "Deterministic selection of short, clear mapped issues where the assigned class is "
    "strongly signaled by the title and opening paragraph; class-specific title/body cues "
    "are preferred and no examples are duplicated."
)
RAG_SELECTION_CRITERIA = (
    "Deterministic selection of mapped issues with at least one maintainer-side comment "
    "from a MEMBER, OWNER, or CONTRIBUTOR containing 80 or more words, with preference for "
    "comments that include code blocks, lists, or numbered steps."
)


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"{display_path(path)}:{line_number} must contain a JSON object")
            records.append(parsed)
    return records


def write_jsonl(records: list[JsonObject], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_json(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{display_path(path)} must contain a JSON object")
    return parsed


def write_json(record: JsonObject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
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


def resolve_repo_path(path: str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else REPO_ROOT / parsed


def metadata_raw_path(metadata: JsonObject) -> Path:
    raw_input_path = metadata.get("raw_input_path")
    if isinstance(raw_input_path, str) and raw_input_path:
        return resolve_repo_path(raw_input_path)
    return DEFAULT_RAW_PATH


def issue_id(record: JsonObject) -> str:
    github_id = record.get("github_id")
    if github_id is not None:
        return str(github_id)
    number = record.get("number")
    if number is not None:
        return str(number)
    raise ValueError("record is missing both github_id and number")


def field_schemas(records: list[JsonObject]) -> list[list[str]]:
    return [list(record.keys()) for record in records[:3]]


def metadata_schema(metadata: JsonObject) -> list[str]:
    return list(metadata.keys())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def title_text(record: JsonObject) -> str:
    text = record.get("text")
    if not isinstance(text, str):
        return ""
    return text.splitlines()[0].strip()


def opening_text(record: JsonObject) -> str:
    text = record.get("text")
    if not isinstance(text, str):
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return "\n\n".join(paragraphs[:2])


def signal_score(label: str, text: str, title: str) -> int:
    text_lower = text.lower()
    title_lower = title.lower()
    cues = {
        "bug": ("bug", "crash", "fail", "failure", "error", "exception", "broken", "regression"),
        "docs": ("doc", "documentation", "readme", "typo", "example", "guide", "link"),
        "feature": ("feature", "add", "support", "allow", "improve", "request", "proposal", "new"),
        "question": ("?", "how", "what", "why", "can i", "could", "help", "question"),
    }
    score = 0
    for cue in cues[label]:
        if cue in title_lower:
            score += 8
        if cue in text_lower:
            score += 3
    if title_lower.startswith((f"{label}:", f"{label} -")):
        score += 20
    return score


def golden_score(record: JsonObject) -> tuple[int, int, int]:
    label = str(record.get("label"))
    title = title_text(record)
    opening = opening_text(record)
    words = word_count(opening)
    shortness = max(0, 220 - words)
    score = signal_score(label, opening, title) + shortness
    return score, -words, -int(record.get("number") or 0)


def select_golden_records(mapped_records: list[JsonObject]) -> list[JsonObject]:
    selected: list[JsonObject] = []
    for label, target_count in GOLDEN_COUNTS.items():
        candidates = [
            record
            for record in mapped_records
            if record.get("label") == label and isinstance(record.get("text"), str)
        ]
        candidates = sorted(candidates, key=golden_score, reverse=True)
        if len(candidates) < target_count:
            raise ValueError(
                f"not enough mapped {label} records for golden set: "
                f"needed {target_count}, found {len(candidates)}"
            )
        selected.extend(candidates[:target_count])
    return selected


def comment_like_field_paths(record: JsonObject) -> list[str]:
    paths: set[str] = set()

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if any(term in key.lower() for term in COMMENT_FIELD_TERMS):
                    paths.add(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:3]):
                visit(child, f"{path}[{index}]")

    visit(record, "")
    return sorted(paths)


def extract_comment_objects(record: JsonObject) -> list[JsonObject]:
    comments: list[JsonObject] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("body"), str) and isinstance(
                value.get("author_association"), str
            ):
                comments.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record)
    return comments


def contains_comment_objects(raw_records: list[JsonObject]) -> bool:
    return any(extract_comment_objects(record) for record in raw_records)


def has_structured_comment_body(body: str) -> bool:
    return (
        "```" in body
        or re.search(r"(?m)^\s*[-*]\s+", body) is not None
        or re.search(r"(?m)^\s*\d+[.)]\s+", body) is not None
    )


def best_rag_comment_score(raw_record: JsonObject) -> int | None:
    best_score: int | None = None
    for comment in extract_comment_objects(raw_record):
        association = str(comment.get("author_association") or "").upper()
        body = comment.get("body")
        if association not in MAINTAINER_ASSOCIATIONS or not isinstance(body, str):
            continue
        words = word_count(body)
        if words < RAG_MIN_WORDS:
            continue
        score = words
        if has_structured_comment_body(body):
            score += 500
        if association in {"MEMBER", "OWNER"}:
            score += 50
        best_score = score if best_score is None else max(best_score, score)
    return best_score


def select_rag_holdout_records(
    mapped_records: list[JsonObject],
    raw_records: list[JsonObject],
    excluded_issue_ids: set[str],
) -> list[JsonObject]:
    raw_by_id = {issue_id(record): record for record in raw_records}
    scored: list[tuple[int, int, JsonObject]] = []
    for mapped_record in mapped_records:
        mapped_issue_id = issue_id(mapped_record)
        if mapped_issue_id in excluded_issue_ids:
            continue
        raw_record = raw_by_id.get(mapped_issue_id)
        if raw_record is None:
            continue
        score = best_rag_comment_score(raw_record)
        if score is None:
            continue
        scored.append((score, int(mapped_record.get("number") or 0), mapped_record))

    scored = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
    if len(scored) < RAG_MIN_COUNT:
        raise ValueError(
            "not enough RAG holdout candidates: "
            f"needed at least {RAG_MIN_COUNT}, found {len(scored)}"
        )
    return [record for _, _, record in scored[:RAG_MAX_COUNT]]


def per_class_counts(records: list[JsonObject]) -> dict[str, int]:
    counts = Counter(str(record.get("label")) for record in records)
    return {label: counts.get(label, 0) for label in LABEL_ORDER}


def assert_disjoint_outputs(named_records: dict[str, list[JsonObject]]) -> None:
    seen: dict[str, str] = {}
    for name, records in named_records.items():
        for record in records:
            record_id = issue_id(record)
            if record_id in seen:
                raise ValueError(f"issue {record_id} appears in both {seen[record_id]} and {name}")
            seen[record_id] = name


def build_summary(
    *,
    status: str,
    mapped_records: list[JsonObject],
    raw_records: list[JsonObject],
    metadata: JsonObject,
    raw_path: Path,
    mapped_sha: str,
    raw_comment_objects_present: bool,
    comment_field_paths: list[str],
    files_written: dict[str, Path],
    output_hashes: dict[str, str],
    output_counts: dict[str, int],
    output_per_class: dict[str, dict[str, int]],
) -> str:
    lines = [
        "reserve carve-out summary",
        f"status: {status}",
        "schemas:",
        f"  {display_path(MAPPED_PATH)} first 3 key lists: {field_schemas(mapped_records)}",
        f"  {display_path(raw_path)} first 3 key lists: {field_schemas(raw_records)}",
        f"  {display_path(METADATA_PATH)} top-level keys: {metadata_schema(metadata)}",
        f"mapped SHA-256 verified: {mapped_sha}",
        f"raw issue comments present: {'yes' if raw_comment_objects_present else 'no'}",
        f"comment-like fields found: {comment_field_paths or 'none'}",
        "files written:",
    ]
    if files_written:
        lines.extend(f"  {name}: {display_path(path)}" for name, path in files_written.items())
    else:
        lines.append("  none")

    lines.append("row counts:")
    if output_counts:
        lines.extend(f"  {name}: {count}" for name, count in output_counts.items())
    else:
        lines.append("  none")

    lines.append("per-class counts:")
    if output_per_class:
        lines.extend(f"  {name}: {counts}" for name, counts in output_per_class.items())
    else:
        lines.append("  none")

    lines.append("SHA-256 hashes:")
    if output_hashes:
        lines.extend(f"  {name}: {file_hash}" for name, file_hash in output_hashes.items())
    else:
        lines.append("  none")
    return "\n".join(lines)


def update_metadata(
    metadata: JsonObject,
    *,
    golden_records: list[JsonObject],
    rag_records: list[JsonObject],
    splittable_records: list[JsonObject],
    output_hashes: dict[str, str],
) -> JsonObject:
    updated = dict(metadata)
    updated["carve_outs"] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "random_seed": RANDOM_SEED,
        "golden_eval": {
            "path": display_path(GOLDEN_PATH),
            "count": len(golden_records),
            "per_class": per_class_counts(golden_records),
            "sha256": output_hashes["golden_eval"],
            "selection_criteria": GOLDEN_SELECTION_CRITERIA,
        },
        "rag_holdout": {
            "path": display_path(RAG_PATH),
            "count": len(rag_records),
            "per_class": per_class_counts(rag_records),
            "sha256": output_hashes["rag_holdout"],
            "selection_criteria": RAG_SELECTION_CRITERIA,
        },
        "splittable_pool": {
            "path": display_path(SPLITTABLE_PATH),
            "count": len(splittable_records),
            "per_class": per_class_counts(splittable_records),
            "sha256": output_hashes["splittable_pool"],
        },
    }
    return updated


def main() -> int:
    random.seed(RANDOM_SEED)

    mapped_records = read_jsonl(MAPPED_PATH)
    metadata = read_json(METADATA_PATH)
    raw_path = metadata_raw_path(metadata)
    raw_records = read_jsonl(raw_path)
    mapped_sha = sha256_file(MAPPED_PATH)
    expected_mapped_sha = metadata.get("mapped_output_sha256")
    comment_field_paths = sorted(
        {path for record in raw_records for path in comment_like_field_paths(record)}
    )
    raw_comment_objects_present = contains_comment_objects(raw_records)

    if mapped_sha != expected_mapped_sha:
        print(
            build_summary(
                status="failed: mapped SHA-256 does not match dataset_metadata.json",
                mapped_records=mapped_records,
                raw_records=raw_records,
                metadata=metadata,
                raw_path=raw_path,
                mapped_sha=mapped_sha,
                raw_comment_objects_present=raw_comment_objects_present,
                comment_field_paths=comment_field_paths,
                files_written={},
                output_hashes={},
                output_counts={},
                output_per_class={},
            )
        )
        return 1

    if not raw_comment_objects_present:
        print(
            build_summary(
                status=(
                    "stopped: raw GitHub issues do not include thread comment objects; "
                    "re-fetch with comments is required for rag_holdout"
                ),
                mapped_records=mapped_records,
                raw_records=raw_records,
                metadata=metadata,
                raw_path=raw_path,
                mapped_sha=mapped_sha,
                raw_comment_objects_present=raw_comment_objects_present,
                comment_field_paths=comment_field_paths,
                files_written={},
                output_hashes={},
                output_counts={},
                output_per_class={},
            )
        )
        return 0

    golden_records = select_golden_records(mapped_records)
    golden_ids = {issue_id(record) for record in golden_records}
    rag_records = select_rag_holdout_records(mapped_records, raw_records, golden_ids)
    reserve_ids = golden_ids | {issue_id(record) for record in rag_records}
    splittable_records = [
        record for record in mapped_records if issue_id(record) not in reserve_ids
    ]

    assert_disjoint_outputs(
        {
            "golden_eval": golden_records,
            "rag_holdout": rag_records,
            "splittable_pool": splittable_records,
        }
    )

    write_jsonl(golden_records, GOLDEN_PATH)
    write_jsonl(rag_records, RAG_PATH)
    write_jsonl(splittable_records, SPLITTABLE_PATH)

    output_hashes = {
        "golden_eval": sha256_file(GOLDEN_PATH),
        "rag_holdout": sha256_file(RAG_PATH),
        "splittable_pool": sha256_file(SPLITTABLE_PATH),
    }
    updated_metadata = update_metadata(
        metadata,
        golden_records=golden_records,
        rag_records=rag_records,
        splittable_records=splittable_records,
        output_hashes=output_hashes,
    )
    write_json(updated_metadata, METADATA_PATH)

    print(
        build_summary(
            status="completed",
            mapped_records=mapped_records,
            raw_records=raw_records,
            metadata=updated_metadata,
            raw_path=raw_path,
            mapped_sha=mapped_sha,
            raw_comment_objects_present=raw_comment_objects_present,
            comment_field_paths=comment_field_paths,
            files_written={
                "golden_eval": GOLDEN_PATH,
                "rag_holdout": RAG_PATH,
                "splittable_pool": SPLITTABLE_PATH,
                "dataset_metadata": METADATA_PATH,
            },
            output_hashes=output_hashes,
            output_counts={
                "golden_eval": len(golden_records),
                "rag_holdout": len(rag_records),
                "splittable_pool": len(splittable_records),
            },
            output_per_class={
                "golden_eval": per_class_counts(golden_records),
                "rag_holdout": per_class_counts(rag_records),
                "splittable_pool": per_class_counts(splittable_records),
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
