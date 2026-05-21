from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPED_PATH = REPO_ROOT / "data" / "processed" / "issues_mapped.jsonl"
DIRTY_BACKUP_PATH = REPO_ROOT / "data" / "processed" / "issues_mapped.dirty.jsonl"
METADATA_PATH = REPO_ROOT / "data" / "processed" / "dataset_metadata.json"
DECISIONS_PATH = REPO_ROOT / "deliverables" / "DECISIONS.md"
CARVE_SCRIPT_PATH = REPO_ROOT / "scripts" / "carve_out_reserves.py"

LABELS = ("bug", "feature", "docs", "question")
CHAR_COUNT_MIN = 20
WORD_COUNT_MIN = 5
JUNK_PATTERNS = [".", ".\n\n.", "..", "n/a", "na", "test", "delete"]
JUNK_PATTERN_SET = {pattern.lower() for pattern in JUNK_PATTERNS}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{display_path(path)} must contain a JSON object")
    return parsed


def write_json(record: dict[str, object], path: Path) -> None:
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


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def has_alnum(text: str) -> bool:
    return re.search(r"[A-Za-z0-9]", text) is not None


def record_quality(record: dict[str, object]) -> dict[str, object]:
    text = record.get("text")
    text_value = text if isinstance(text, str) else ""
    stripped = text_value.strip()
    return {
        "char_count": len(stripped),
        "word_count": word_count(stripped),
        "is_only_punct": bool(stripped) and not has_alnum(stripped),
        "stripped": stripped,
    }


def percentile(sorted_values: list[int], percent: int) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = round((len(sorted_values) - 1) * percent / 100)
    return sorted_values[position]


def distribution(values: list[int]) -> dict[str, int]:
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0] if sorted_values else 0,
        "p5": percentile(sorted_values, 5),
        "p25": percentile(sorted_values, 25),
        "median": percentile(sorted_values, 50),
        "p75": percentile(sorted_values, 75),
        "p95": percentile(sorted_values, 95),
        "max": sorted_values[-1] if sorted_values else 0,
    }


def empty_class_lists() -> dict[str, list[int]]:
    return {label: [] for label in LABELS}


def per_class_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        label = record.get("label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts


def compute_distributions(
    records: list[dict[str, object]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    char_counts = empty_class_lists()
    word_counts = empty_class_lists()
    for record in records:
        label = record.get("label")
        if not isinstance(label, str):
            continue
        quality = record_quality(record)
        char_counts.setdefault(label, []).append(int(quality["char_count"]))
        word_counts.setdefault(label, []).append(int(quality["word_count"]))
    return (
        {label: distribution(char_counts.get(label, [])) for label in LABELS},
        {label: distribution(word_counts.get(label, [])) for label in LABELS},
    )


def should_drop(quality: dict[str, object]) -> bool:
    stripped = str(quality["stripped"])
    return (
        int(quality["char_count"]) < CHAR_COUNT_MIN
        or int(quality["word_count"]) < WORD_COUNT_MIN
        or bool(quality["is_only_punct"])
        or stripped.lower() in JUNK_PATTERN_SET
    )


def first_80(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:80]


def clean_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    cleaned: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    dropped_per_class = {label: 0 for label in LABELS}
    for record in records:
        quality = record_quality(record)
        if should_drop(quality):
            label = (
                record.get("label")
                if isinstance(record.get("label"), str)
                else "(missing)"
            )
            dropped_per_class[label] = dropped_per_class.get(label, 0) + 1
            dropped.append(
                {
                    "github_id": record.get("github_id"),
                    "label": label,
                    "char_count": quality["char_count"],
                    "word_count": quality["word_count"],
                    "text_preview": first_80(str(quality["stripped"])),
                }
            )
        else:
            cleaned.append(record)
    return cleaned, dropped, dropped_per_class


def update_metadata(
    *,
    metadata: dict[str, object],
    cleaned_records: list[dict[str, object]],
    mapped_sha256: str,
    dropped_count: int,
    dropped_per_class: dict[str, int],
) -> None:
    metadata["mapped_issue_count"] = len(cleaned_records)
    metadata["mapped_per_class"] = per_class_counts(cleaned_records)
    metadata["mapped_output_sha256"] = mapped_sha256
    metadata["quality_filters_applied"] = {
        "char_count_min": CHAR_COUNT_MIN,
        "word_count_min": WORD_COUNT_MIN,
        "drop_only_punct": True,
        "junk_patterns_dropped": JUNK_PATTERNS,
        "dropped_count": dropped_count,
        "dropped_per_class": dropped_per_class,
    }
    if "carve_outs" in metadata:
        del metadata["carve_outs"]


def run_carve_out_reserves() -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - local deterministic script, no user input.
        [sys.executable, str(CARVE_SCRIPT_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def append_decision(
    *,
    rows_in: int,
    rows_out: int,
    dropped_per_class: dict[str, int],
) -> None:
    existing = DECISIONS_PATH.read_text(encoding="utf-8")
    marker = "## D1.17 Data-quality cleaning pass"
    if marker in existing:
        start = existing.index(marker)
        next_day = existing.find("\n# Day 2 decisions", start)
        if next_day == -1:
            existing = existing[:start].rstrip() + "\n\n"
        else:
            existing = (
                existing[:start].rstrip() + "\n\n" + existing[next_day:].lstrip("\n")
            )

    decision = f"""{marker}

**Decision:** Apply a deterministic quality filter to the mapped classifier dataset before reserve
carving.

**Filter rules and rationale:**

- Drop `char_count < {CHAR_COUNT_MIN}` because shorter records do not carry enough context for a
  useful issue classifier example.
- Drop `word_count < {WORD_COUNT_MIN}` because fewer than five lexical tokens usually means a
  placeholder, fragment, or accidental issue body.
- Drop only-punctuation text because it has no alphanumeric signal for label learning or evaluation.
- Drop obvious junk patterns (`.`, `.\\n\\n.`, `..`, `n/a`, `na`, `test`, `delete`) because
  they are explicit placeholders rather than maintainer-triage examples.

**Counts:**

- Rows in: {rows_in}.
- Rows out: {rows_out}.
- Dropped per class: {json.dumps(dropped_per_class, sort_keys=True)}.

`data/processed/issues_mapped.dirty.jsonl` preserves the pre-cleaning mapped dataset for
auditability.

"""
    day_2_marker = "# Day 2 decisions"
    if day_2_marker in existing:
        insertion_index = existing.index(day_2_marker)
        updated = (
            existing[:insertion_index].rstrip()
            + "\n\n"
            + decision
            + existing[insertion_index:]
        )
    else:
        updated = existing.rstrip() + "\n\n" + decision
    DECISIONS_PATH.write_text(updated, encoding="utf-8")


def load_carve_hashes(metadata: dict[str, object]) -> dict[str, str]:
    carve_outs = metadata.get("carve_outs")
    if not isinstance(carve_outs, dict):
        raise ValueError(
            "dataset_metadata.json does not contain regenerated carve_outs"
        )

    hashes: dict[str, str] = {}
    for key in ["golden_eval", "rag_holdout", "splittable_pool"]:
        section = carve_outs.get(key)
        if not isinstance(section, dict) or not isinstance(section.get("sha256"), str):
            raise ValueError(f"carve_outs.{key}.sha256 is missing")
        hashes[key] = str(section["sha256"])
    return hashes


def build_output(
    *,
    char_distributions: dict[str, dict[str, int]],
    word_distributions: dict[str, dict[str, int]],
    dropped: list[dict[str, object]],
    dropped_per_class: dict[str, int],
    rows_in: int,
    rows_out: int,
    mapped_sha256: str,
    carve_hashes: dict[str, str],
) -> str:
    lines = ["data-quality cleaning summary", "char_count distributions:"]
    for label in LABELS:
        lines.append(f"  {label}: {char_distributions[label]}")
    lines.append("word_count distributions:")
    for label in LABELS:
        lines.append(f"  {label}: {word_distributions[label]}")
    lines.append("dropped rows:")
    if dropped:
        for row in dropped:
            lines.append(
                "  "
                f"github_id={row['github_id']} "
                f"label={row['label']} "
                f"char_count={row['char_count']} "
                f"word_count={row['word_count']} "
                f"text={json.dumps(row['text_preview'], ensure_ascii=False)}"
            )
    else:
        lines.append("  none")
    lines.append(f"dropped per class: {dropped_per_class}")
    lines.append(
        "cleaning: "
        f"{rows_in} -> {rows_out}  "
        f"(dropped {rows_in - rows_out}: "
        f"bug={dropped_per_class.get('bug', 0)}, "
        f"feature={dropped_per_class.get('feature', 0)}, "
        f"docs={dropped_per_class.get('docs', 0)}, "
        f"question={dropped_per_class.get('question', 0)})"
    )
    lines.append(f"new mapped sha256: {mapped_sha256}")
    lines.append("carve-out subprocess: ok")
    lines.append(f"new golden_eval sha256: {carve_hashes['golden_eval']}")
    lines.append(f"new rag_holdout sha256: {carve_hashes['rag_holdout']}")
    lines.append(f"new splittable_pool sha256: {carve_hashes['splittable_pool']}")
    return "\n".join(lines)


def main() -> int:
    records = read_jsonl(MAPPED_PATH)
    char_distributions, word_distributions = compute_distributions(records)
    cleaned_records, dropped, dropped_per_class = clean_records(records)

    if not DIRTY_BACKUP_PATH.exists():
        shutil.copy2(MAPPED_PATH, DIRTY_BACKUP_PATH)

    write_jsonl(cleaned_records, MAPPED_PATH)
    mapped_sha256 = sha256_file(MAPPED_PATH)
    metadata = read_json(METADATA_PATH)
    update_metadata(
        metadata=metadata,
        cleaned_records=cleaned_records,
        mapped_sha256=mapped_sha256,
        dropped_count=len(dropped),
        dropped_per_class=dropped_per_class,
    )
    write_json(metadata, METADATA_PATH)

    carve_result = run_carve_out_reserves()
    if carve_result.returncode != 0:
        print(carve_result.stdout.strip())
        print(carve_result.stderr.strip())
        return carve_result.returncode

    metadata = read_json(METADATA_PATH)
    carve_hashes = load_carve_hashes(metadata)
    append_decision(
        rows_in=len(records),
        rows_out=len(cleaned_records),
        dropped_per_class=dropped_per_class,
    )
    print(
        build_output(
            char_distributions=char_distributions,
            word_distributions=word_distributions,
            dropped=dropped,
            dropped_per_class=dropped_per_class,
            rows_in=len(records),
            rows_out=len(cleaned_records),
            mapped_sha256=mapped_sha256,
            carve_hashes=carve_hashes,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
