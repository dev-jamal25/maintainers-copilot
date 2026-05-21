from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "github_issues.jsonl"
DEFAULT_MAPPING_PATH = REPO_ROOT / "ml" / "classification" / "label_mapping.yaml"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_MAPPED_OUTPUT_NAME = "issues_mapped.jsonl"
STALE_SPLIT_OUTPUT_NAMES = (
    "issues_train.jsonl",
    "issues_val.jsonl",
    "issues_test.jsonl",
)
TARGET_ISSUES_PER_CLASS = 500
TARGET_LABELS = {"bug", "feature", "docs", "question"}


@dataclass(frozen=True)
class LabelMapping:
    repo: str
    target_labels: dict[str, int]
    github_label_mapping: dict[str, list[str]]
    priority: list[str]
    reverse_mapping: dict[str, list[str]]
    raw_mapping: JsonObject


@dataclass(frozen=True)
class ProcessedDataset:
    records: list[JsonObject]
    metadata: JsonObject


def load_local_env() -> None:
    """Load repo-root .env explicitly before reading dataset env vars."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def load_label_mapping(path: Path = DEFAULT_MAPPING_PATH) -> LabelMapping:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Label mapping must be a mapping: {path}")

    repo = raw.get("repo")
    target_labels = raw.get("target_labels")
    github_label_mapping = raw.get("github_label_mapping")
    if not isinstance(repo, str):
        raise ValueError("label_mapping.yaml must define repo")
    if not isinstance(target_labels, dict):
        raise ValueError("label_mapping.yaml must define target_labels")
    if not isinstance(github_label_mapping, dict):
        raise ValueError("label_mapping.yaml must define github_label_mapping")

    parsed_target_labels: dict[str, int] = {}
    for target, label_id in target_labels.items():
        if not isinstance(target, str) or not isinstance(label_id, int):
            raise ValueError("target_labels must map strings to integer IDs")
        parsed_target_labels[target] = label_id

    if set(parsed_target_labels) != TARGET_LABELS:
        raise ValueError(f"target_labels must be exactly {sorted(TARGET_LABELS)}")

    parsed_github_mapping: dict[str, list[str]] = {}
    reverse_mapping: dict[str, list[str]] = defaultdict(list)
    for target, labels in github_label_mapping.items():
        if target not in parsed_target_labels:
            raise ValueError(f"github_label_mapping contains unknown target: {target}")
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(f"github_label_mapping.{target} must be a list of strings")

        normalized_labels = [normalize_label(label) for label in labels]
        parsed_github_mapping[target] = normalized_labels
        for normalized_label in normalized_labels:
            reverse_mapping[normalized_label].append(target)

    priority = list(parsed_target_labels)
    return LabelMapping(
        repo=repo,
        target_labels=parsed_target_labels,
        github_label_mapping=parsed_github_mapping,
        priority=priority,
        reverse_mapping=dict(reverse_mapping),
        raw_mapping=raw,
    )


def extract_issue_label_names(raw_labels: object) -> list[str]:
    if not isinstance(raw_labels, list):
        return []

    names: list[str] = []
    for label in raw_labels:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str):
                names.append(name)
        elif isinstance(label, str):
            names.append(label)
    return names


def map_issue_labels(
    raw_labels: object,
    mapping: LabelMapping,
) -> tuple[str | None, int | None, list[str]]:
    matched_targets: set[str] = set()
    for label in extract_issue_label_names(raw_labels):
        for target in mapping.reverse_mapping.get(normalize_label(label), []):
            matched_targets.add(target)

    if not matched_targets:
        return None, None, []

    ordered_matches = [
        target for target in mapping.priority if target in matched_targets
    ]
    chosen_label = ordered_matches[0]
    return chosen_label, mapping.target_labels[chosen_label], ordered_matches


def map_issue_to_label(
    issue: JsonObject,
    mapping: LabelMapping,
) -> tuple[str | None, int | None, list[str], str | None]:
    fetched_for_label = issue.get("fetched_for_label")
    fetched_for_github_label = issue.get("fetched_for_github_label")
    normalized_issue_labels = {
        normalize_label(label)
        for label in extract_issue_label_names(issue.get("labels"))
    }

    if isinstance(fetched_for_label, str) and fetched_for_label:
        if fetched_for_label not in mapping.target_labels:
            return None, None, [], "invalid_fetched_for_label"

        mapped_github_labels = set(mapping.github_label_mapping[fetched_for_label])
        if isinstance(fetched_for_github_label, str) and fetched_for_github_label:
            normalized_fetch_label = normalize_label(fetched_for_github_label)
            if normalized_fetch_label not in mapped_github_labels:
                return None, None, [], "invalid_fetched_for_github_label"
            if normalized_fetch_label not in normalized_issue_labels:
                return None, None, [], "missing_fetched_github_label"
        elif normalized_issue_labels.isdisjoint(mapped_github_labels):
            return None, None, [], "missing_fetched_github_label"

        return (
            fetched_for_label,
            mapping.target_labels[fetched_for_label],
            [fetched_for_label],
            None,
        )

    label, label_id, matched_labels = map_issue_labels(issue.get("labels"), mapping)
    return label, label_id, matched_labels, None


def clean_text(title: object, body: object) -> str | None:
    title_text = title.strip() if isinstance(title, str) else ""
    body_text = body.strip() if isinstance(body, str) else ""
    text = "\n\n".join(part for part in [title_text, body_text] if part)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        return None
    return text


def parse_github_datetime(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty GitHub timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(parsed)
    return records


def write_jsonl(records: list[JsonObject], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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


def output_record(
    issue: JsonObject, label: str, label_id: int, text: str
) -> JsonObject:
    return {
        "github_id": issue.get("github_id"),
        "number": issue.get("number"),
        "html_url": issue.get("html_url"),
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "fetched_for_label": issue.get("fetched_for_label"),
        "fetched_for_github_label": issue.get("fetched_for_github_label"),
        "label": label,
        "label_id": label_id,
        "text": text,
    }


def build_mapped_records(
    raw_issues: list[JsonObject], mapping: LabelMapping
) -> ProcessedDataset:
    records: list[JsonObject] = []
    skipped_by_reason: Counter[str] = Counter()
    unmatched_label_stats: Counter[str] = Counter()
    multi_label_examples: list[JsonObject] = []
    usable_class_counts: Counter[str] = Counter()

    for issue in raw_issues:
        try:
            parse_github_datetime(issue.get("closed_at"))
        except ValueError:
            skipped_by_reason["missing_closed_at"] += 1
            continue

        label, label_id, matched_labels, mapping_skip_reason = map_issue_to_label(
            issue, mapping
        )
        if label is None or label_id is None:
            skipped_by_reason[mapping_skip_reason or "unmapped_labels"] += 1
            label_names = extract_issue_label_names(issue.get("labels"))
            if label_names:
                unmatched_label_stats.update(
                    normalize_label(name) for name in label_names
                )
            else:
                unmatched_label_stats["(no labels)"] += 1
            continue

        text = clean_text(issue.get("title"), issue.get("body"))
        if text is None:
            skipped_by_reason["empty_title_and_body"] += 1
            continue

        if len(matched_labels) > 1:
            skipped_by_reason["multi_label_conflicts_resolved"] += 1
            if len(multi_label_examples) < 20:
                multi_label_examples.append(
                    {
                        "number": issue.get("number"),
                        "matched_labels": matched_labels,
                        "chosen_label": label,
                    }
                )

        record = output_record(issue, label, label_id, text)
        records.append(record)
        usable_class_counts[label] += 1

    per_class_shortfalls: dict[str, JsonObject] = {}
    for label in mapping.priority:
        usable_count = usable_class_counts.get(label, 0)
        if usable_count < TARGET_ISSUES_PER_CLASS:
            per_class_shortfalls[label] = {
                "target": TARGET_ISSUES_PER_CLASS,
                "usable": usable_count,
                "missing": TARGET_ISSUES_PER_CLASS - usable_count,
            }

    records = sorted(
        records, key=lambda record: parse_github_datetime(record["closed_at"])
    )

    metadata: JsonObject = {
        "raw_issue_count": len(raw_issues),
        "mapped_issue_count": len(records),
        "skipped_issue_count": len(raw_issues) - len(records),
        "target_issues_per_class": TARGET_ISSUES_PER_CLASS,
        "target_total_issues": TARGET_ISSUES_PER_CLASS * len(mapping.priority),
        "mapped_per_class": {
            label: usable_class_counts.get(label, 0) for label in mapping.priority
        },
        "per_class_shortfalls": per_class_shortfalls,
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "unmatched_label_stats": dict(unmatched_label_stats.most_common()),
        "multi_label_conflict_examples": multi_label_examples,
    }
    return ProcessedDataset(records=records, metadata=metadata)


def build_metadata(
    *,
    mapping: LabelMapping,
    raw_path: Path,
    mapped_output_path: Path,
    processing_metadata: JsonObject,
) -> JsonObject:
    return {
        "repo": mapping.repo,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "raw_input_path": display_path(raw_path),
        "raw_input_sha256": sha256_file(raw_path),
        "mapped_output_path": display_path(mapped_output_path),
        "mapped_output_sha256": sha256_file(mapped_output_path),
        **processing_metadata,
        "label_mapping_used": mapping.raw_mapping,
    }


def write_metadata(metadata: JsonObject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def print_export_summary(metadata: JsonObject, metadata_path: Path) -> None:
    print(f"repo: {metadata['repo']}")
    print(f"raw issue count: {metadata['raw_issue_count']}")
    print(f"mapped issue count: {metadata['mapped_issue_count']}")
    print(f"skipped issue count: {metadata['skipped_issue_count']}")
    print(f"target issues per class: {metadata['target_issues_per_class']}")
    print(f"mapped per class: {metadata['mapped_per_class']}")
    print(f"mapped dataset path: {metadata['mapped_output_path']}")
    print(f"metadata path: {display_path(metadata_path)}")

    raw_issue_count = int(metadata["raw_issue_count"])
    mapped_issue_count = int(metadata["mapped_issue_count"])
    if raw_issue_count and (
        mapped_issue_count < 100 or mapped_issue_count / raw_issue_count < 0.2
    ):
        print(
            "WARNING: mapped issue count is low. Inspect unmatched_label_stats in "
            f"{metadata_path} and update ml/classification/label_mapping.yaml if needed."
        )
    if metadata["per_class_shortfalls"]:
        print(
            "WARNING: one or more classes has fewer than the 500 usable issue target. "
            "No examples were duplicated or faked; inspect per_class_shortfalls in "
            f"{metadata_path}."
        )


def remove_stale_split_outputs(output_dir: Path) -> None:
    for output_name in STALE_SPLIT_OUTPUT_NAMES:
        stale_path = output_dir / output_name
        if stale_path.exists():
            stale_path.unlink()


def build_dataset(
    *,
    raw_path: Path = DEFAULT_RAW_PATH,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
) -> JsonObject:
    load_local_env()
    mapping = load_label_mapping(mapping_path)
    env_repo = os.getenv("GITHUB_REPO")
    if env_repo and env_repo != mapping.repo:
        print(
            "WARNING: GITHUB_REPO differs from label_mapping.yaml repo. "
            f"Using mapping repo {mapping.repo} for metadata."
        )

    raw_issues = read_jsonl(raw_path)
    processed = build_mapped_records(raw_issues, mapping)

    mapped_output_path = output_dir / DEFAULT_MAPPED_OUTPUT_NAME
    write_jsonl(processed.records, mapped_output_path)
    remove_stale_split_outputs(output_dir)

    metadata = build_metadata(
        mapping=mapping,
        raw_path=raw_path,
        mapped_output_path=mapped_output_path,
        processing_metadata=processed.metadata,
    )
    metadata_path = output_dir / "dataset_metadata.json"
    write_metadata(metadata, metadata_path)
    print_export_summary(metadata, metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map raw GitHub issues and write one classifier dataset JSONL."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    build_dataset(
        raw_path=resolve_repo_path(args.raw_path),
        mapping_path=resolve_repo_path(args.mapping_path),
        output_dir=resolve_repo_path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
