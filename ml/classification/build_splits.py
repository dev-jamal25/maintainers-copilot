from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

JsonObject = dict[str, Any]

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_PATH = REPO_ROOT / "data" / "splits" / "splittable_pool.jsonl"
DEFAULT_GOLDEN_PATH = REPO_ROOT / "data" / "splits" / "golden_eval.jsonl"
DEFAULT_RAG_PATH = REPO_ROOT / "data" / "splits" / "rag_holdout.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "processed" / "dataset_metadata.json"

ALLOWED_LABELS = frozenset({"bug", "feature", "docs", "question"})
LABEL_ORDER = ("bug", "feature", "docs", "question")
REQUIRED_FIELDS = (
    "github_id",
    "number",
    "html_url",
    "created_at",
    "closed_at",
    "fetched_for_label",
    "fetched_for_github_label",
    "label",
    "label_id",
    "text",
)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42
TIMESTAMP_FIELD = "created_at"
TEMPORAL_POLICY = "per_class_created_at"


class SplitError(ValueError):
    """Raised when classification split generation cannot safely continue."""


@dataclass(frozen=True)
class SplitRecord:
    issue_id: str
    label: str
    created_at: datetime
    raw: JsonObject
    original_index: int


@dataclass(frozen=True)
class SplitPaths:
    pool: Path = DEFAULT_POOL_PATH
    golden: Path = DEFAULT_GOLDEN_PATH
    rag: Path = DEFAULT_RAG_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    metadata: Path = DEFAULT_METADATA_PATH


@dataclass(frozen=True)
class SplitSummary:
    input_count: int
    split_counts: dict[str, int]
    per_class_counts: dict[str, dict[str, int]]
    timestamp_boundaries: dict[str, dict[str, str | None]]
    output_hashes: dict[str, str]
    output_paths: dict[str, Path]
    metadata_path: Path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed: object = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SplitError(
                        f"{display_path(path)}:{line_number} contains invalid JSON"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise SplitError(f"{display_path(path)}:{line_number} must be a JSON object")
                records.append(cast(JsonObject, parsed))
    except OSError as exc:
        raise SplitError(f"could not read {display_path(path)}: {exc}") from exc
    return records


def write_jsonl(records: list[SplitRecord], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.raw, ensure_ascii=False))
                f.write("\n")
    except OSError as exc:
        raise SplitError(f"could not write {display_path(path)}: {exc}") from exc


def read_metadata(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            parsed: object = json.load(f)
    except json.JSONDecodeError as exc:
        raise SplitError(f"{display_path(path)} contains invalid JSON") from exc
    except OSError as exc:
        raise SplitError(f"could not read {display_path(path)}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SplitError(f"{display_path(path)} must contain a JSON object")
    return cast(JsonObject, parsed)


def write_metadata(metadata: JsonObject, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        raise SplitError(f"could not write {display_path(path)}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SplitError(f"could not hash {display_path(path)}: {exc}") from exc
    return digest.hexdigest()


def issue_id(record: JsonObject, *, path: Path, line_number: int) -> str:
    github_id = record.get("github_id")
    if github_id is not None:
        return str(github_id)
    number = record.get("number")
    if number is not None:
        return str(number)
    raise SplitError(f"{display_path(path)}:{line_number} is missing github_id and number")


def validate_required_fields(record: JsonObject, *, path: Path, line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record or record[field] is None]
    if missing:
        fields = ", ".join(missing)
        raise SplitError(f"{display_path(path)}:{line_number} missing required field(s): {fields}")


def require_string_field(record: JsonObject, field: str, *, path: Path, line_number: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise SplitError(f"{display_path(path)}:{line_number} field {field!r} must be a string")
    return value


def parse_created_at(value: str, *, path: Path, line_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SplitError(
            f"{display_path(path)}:{line_number} field {TIMESTAMP_FIELD!r} is not parseable"
        ) from exc
    if parsed.tzinfo is None:
        raise SplitError(
            f"{display_path(path)}:{line_number} field {TIMESTAMP_FIELD!r} must include timezone"
        )
    return parsed


def load_records(path: Path) -> list[SplitRecord]:
    raw_records = read_jsonl(path)
    records: list[SplitRecord] = []
    for index, raw in enumerate(raw_records):
        line_number = index + 1
        validate_required_fields(raw, path=path, line_number=line_number)

        label = require_string_field(raw, "label", path=path, line_number=line_number)
        if label not in ALLOWED_LABELS:
            allowed = ", ".join(LABEL_ORDER)
            raise SplitError(
                f"{display_path(path)}:{line_number} has invalid label {label!r}; "
                f"allowed labels: {allowed}"
            )

        require_string_field(raw, "html_url", path=path, line_number=line_number)
        require_string_field(raw, "closed_at", path=path, line_number=line_number)
        require_string_field(raw, "fetched_for_label", path=path, line_number=line_number)
        require_string_field(raw, "fetched_for_github_label", path=path, line_number=line_number)
        require_string_field(raw, "text", path=path, line_number=line_number)
        created_at = parse_created_at(
            require_string_field(raw, TIMESTAMP_FIELD, path=path, line_number=line_number),
            path=path,
            line_number=line_number,
        )
        label_id = raw["label_id"]
        if not isinstance(label_id, int) or isinstance(label_id, bool):
            raise SplitError(f"{display_path(path)}:{line_number} field 'label_id' must be an int")

        records.append(
            SplitRecord(
                issue_id=issue_id(raw, path=path, line_number=line_number),
                label=label,
                created_at=created_at,
                raw=raw,
                original_index=index,
            )
        )

    if not records:
        raise SplitError(f"{display_path(path)} does not contain any records")
    LOGGER.info("Loaded %s splittable records from %s", len(records), display_path(path))
    return records


def load_reserve_ids(*paths: Path) -> set[str]:
    reserve_ids: set[str] = set()
    for path in paths:
        records = read_jsonl(path)
        for index, record in enumerate(records):
            reserve_ids.add(issue_id(record, path=path, line_number=index + 1))
        LOGGER.info("Loaded %s reserve ids from %s", len(records), display_path(path))
    return reserve_ids


def validate_unique_ids(records: list[SplitRecord]) -> None:
    counts = Counter(record.issue_id for record in records)
    duplicate_ids = sorted(issue for issue, count in counts.items() if count > 1)
    if duplicate_ids:
        sample = ", ".join(duplicate_ids[:10])
        raise SplitError(f"splittable pool contains duplicate issue id(s): {sample}")


def validate_no_overlap(pool: list[SplitRecord], reserve_ids: set[str]) -> None:
    overlapping = sorted({record.issue_id for record in pool} & reserve_ids)
    if overlapping:
        sample = ", ".join(overlapping[:10])
        raise SplitError(f"splittable pool overlaps reserved eval/RAG issue id(s): {sample}")


def temporal_key(record: SplitRecord) -> tuple[datetime, int]:
    return record.created_at, record.original_index


def split_train_val(
    records: list[SplitRecord], *, seed: int
) -> tuple[list[SplitRecord], list[SplitRecord]]:
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train: list[SplitRecord] = []
    val: list[SplitRecord] = []

    for label_index, label in enumerate(LABEL_ORDER):
        group = [record for record in records if record.label == label]
        shuffled = sorted(group, key=temporal_key)
        random.Random(seed + label_index).shuffle(shuffled)

        val_count = round(len(shuffled) * val_fraction)
        if len(shuffled) > 1:
            val_count = min(max(val_count, 1), len(shuffled) - 1)
        else:
            val_count = 0

        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])

    return sorted(train, key=temporal_key), sorted(val, key=temporal_key)


def split_label_by_time(
    records: list[SplitRecord], *, label: str
) -> tuple[list[SplitRecord], list[SplitRecord], int]:
    sorted_records = sorted(records, key=temporal_key)
    target_test_count = max(1, round(len(sorted_records) * TEST_RATIO))
    boundary_index = len(sorted_records) - target_test_count
    if boundary_index <= 0:
        raise SplitError(f"not enough {label!r} records to carve out a temporal test split")

    cutoff = sorted_records[boundary_index].created_at
    older_pool = [record for record in sorted_records if record.created_at < cutoff]
    test = [record for record in sorted_records if record.created_at >= cutoff]

    if len(older_pool) < 2 or not test:
        raise SplitError(
            f"timestamp tie handling for {label!r} would prevent non-empty "
            "train, val, and test splits"
        )
    return older_pool, test, target_test_count


def split_by_time(
    records: list[SplitRecord], *, seed: int
) -> tuple[list[SplitRecord], list[SplitRecord], list[SplitRecord], int]:
    older_pool: list[SplitRecord] = []
    test: list[SplitRecord] = []
    target_test_count = 0
    for label in LABEL_ORDER:
        label_records = [record for record in records if record.label == label]
        if not label_records:
            raise SplitError(f"splittable pool does not contain any {label!r} records")
        label_older_pool, label_test, label_target_test_count = split_label_by_time(
            label_records, label=label
        )
        older_pool.extend(label_older_pool)
        test.extend(label_test)
        target_test_count += label_target_test_count

    train, val = split_train_val(older_pool, seed=seed)
    validate_all_labels_present({"train": train, "val": val, "test": test})
    validate_per_class_temporal_boundary(train=train, test=test)
    validate_per_class_temporal_boundary(train=val, test=test)
    LOGGER.info(
        "Split %s records into train=%s val=%s test=%s",
        len(records),
        len(train),
        len(val),
        len(test),
    )
    return train, val, test, target_test_count


def validate_all_labels_present(named_records: dict[str, list[SplitRecord]]) -> None:
    missing: list[str] = []
    for split_name, records in named_records.items():
        counts = per_class_counts(records)
        for label in LABEL_ORDER:
            if counts[label] == 0:
                missing.append(f"{split_name}:{label}")
    if missing:
        raise SplitError(f"each label must appear in every split; missing {', '.join(missing)}")


def validate_temporal_boundary(*, train: list[SplitRecord], test: list[SplitRecord]) -> None:
    if not train or not test:
        raise SplitError("train/val and test splits must be non-empty")
    latest_train = max(record.created_at for record in train)
    earliest_test = min(record.created_at for record in test)
    if latest_train >= earliest_test:
        raise SplitError(
            "temporal invariant failed: "
            f"max(train.created_at)={format_datetime(latest_train)} must be < "
            f"min(test.created_at)={format_datetime(earliest_test)}"
        )


def validate_per_class_temporal_boundary(
    *, train: list[SplitRecord], test: list[SplitRecord]
) -> None:
    for label in LABEL_ORDER:
        label_train = [record for record in train if record.label == label]
        label_test = [record for record in test if record.label == label]
        validate_temporal_boundary(train=label_train, test=label_test)


def split_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "train": output_dir / "classification_train.jsonl",
        "val": output_dir / "classification_val.jsonl",
        "test": output_dir / "classification_test.jsonl",
    }


def format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def per_class_counts(records: list[SplitRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {label: counts.get(label, 0) for label in LABEL_ORDER}


def timestamp_bounds(records: list[SplitRecord]) -> dict[str, str | None]:
    if not records:
        return {"min_created_at": None, "max_created_at": None}
    timestamps = [record.created_at for record in records]
    return {
        "min_created_at": format_datetime(min(timestamps)),
        "max_created_at": format_datetime(max(timestamps)),
    }


def per_class_timestamp_bounds(records: list[SplitRecord]) -> dict[str, dict[str, str | None]]:
    return {
        label: timestamp_bounds([record for record in records if record.label == label])
        for label in LABEL_ORDER
    }


def split_metadata(
    *,
    records: list[SplitRecord],
    path: Path,
    file_hash: str,
) -> JsonObject:
    return {
        "count": len(records),
        "max_created_at": timestamp_bounds(records)["max_created_at"],
        "min_created_at": timestamp_bounds(records)["min_created_at"],
        "path": display_path(path),
        "per_class": per_class_counts(records),
        "per_class_created_at": per_class_timestamp_bounds(records),
        "sha256": file_hash,
    }


def reserve_metadata(path: Path) -> JsonObject:
    records = read_jsonl(path)
    return {
        "count": len(records),
        "path": display_path(path),
        "sha256": sha256_file(path),
    }


def build_metadata_block(
    *,
    paths: SplitPaths,
    pool_records: list[SplitRecord],
    train: list[SplitRecord],
    val: list[SplitRecord],
    test: list[SplitRecord],
    output_paths: dict[str, Path],
    output_hashes: dict[str, str],
    seed: int,
    target_test_count: int,
) -> JsonObject:
    split_counts = {"train": len(train), "val": len(val), "test": len(test)}
    total = sum(split_counts.values())
    return {
        "actual_ratios": {name: split_counts[name] / total for name in ("train", "val", "test")},
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "input_pool": {
            "count": len(pool_records),
            "max_created_at": timestamp_bounds(pool_records)["max_created_at"],
            "min_created_at": timestamp_bounds(pool_records)["min_created_at"],
            "path": display_path(paths.pool),
            "per_class": per_class_counts(pool_records),
            "sha256": sha256_file(paths.pool),
        },
        "random_seed": seed,
        "requested_ratios": {"test": TEST_RATIO, "train": TRAIN_RATIO, "val": VAL_RATIO},
        "reserves_excluded": {
            "golden_eval": reserve_metadata(paths.golden),
            "rag_holdout": reserve_metadata(paths.rag),
        },
        "split_policy": {
            "temporal_policy": TEMPORAL_POLICY,
            "test_selection": "newest records by created_at within each label",
            "target_test_count": target_test_count,
            "tie_policy": (
                "If a per-label target boundary cuts through equal created_at timestamps, all "
                "records at that timestamp are assigned to test so train/val remain strictly "
                "older for that label."
            ),
            "train_val_selection": "fixed-seed stratified shuffle within each label's older pool",
        },
        "splits": {
            "test": split_metadata(
                records=test,
                path=output_paths["test"],
                file_hash=output_hashes["test"],
            ),
            "train": split_metadata(
                records=train,
                path=output_paths["train"],
                file_hash=output_hashes["train"],
            ),
            "val": split_metadata(
                records=val,
                path=output_paths["val"],
                file_hash=output_hashes["val"],
            ),
        },
        "timestamp_field": TIMESTAMP_FIELD,
    }


def update_metadata(metadata_path: Path, block: JsonObject) -> None:
    metadata = read_metadata(metadata_path)
    metadata["classification_splits"] = block
    write_metadata(metadata, metadata_path)


def build_splits(paths: SplitPaths | None = None, *, seed: int = RANDOM_SEED) -> SplitSummary:
    if paths is None:
        paths = SplitPaths()

    pool_records = load_records(paths.pool)
    validate_unique_ids(pool_records)
    reserve_ids = load_reserve_ids(paths.golden, paths.rag)
    validate_no_overlap(pool_records, reserve_ids)

    train, val, test, target_test_count = split_by_time(pool_records, seed=seed)
    output_paths = split_output_paths(paths.output_dir)
    write_jsonl(train, output_paths["train"])
    write_jsonl(val, output_paths["val"])
    write_jsonl(test, output_paths["test"])

    output_hashes = {name: sha256_file(path) for name, path in output_paths.items()}
    metadata_block = build_metadata_block(
        paths=paths,
        pool_records=pool_records,
        train=train,
        val=val,
        test=test,
        output_paths=output_paths,
        output_hashes=output_hashes,
        seed=seed,
        target_test_count=target_test_count,
    )
    update_metadata(paths.metadata, metadata_block)

    per_class = {
        "test": per_class_counts(test),
        "train": per_class_counts(train),
        "val": per_class_counts(val),
    }
    boundaries = {
        "test": timestamp_bounds(test),
        "train": timestamp_bounds(train),
        "val": timestamp_bounds(val),
    }
    summary = SplitSummary(
        input_count=len(pool_records),
        split_counts={"test": len(test), "train": len(train), "val": len(val)},
        per_class_counts=per_class,
        timestamp_boundaries=boundaries,
        output_hashes=output_hashes,
        output_paths=output_paths,
        metadata_path=paths.metadata,
    )
    LOGGER.info("Input rows: %s", summary.input_count)
    LOGGER.info("Split counts: %s", summary.split_counts)
    LOGGER.info("Per-class counts: %s", summary.per_class_counts)
    LOGGER.info("Timestamp boundaries: %s", summary.timestamp_boundaries)
    LOGGER.info("Output hashes: %s", summary.output_hashes)
    LOGGER.info("Metadata updated: %s", display_path(paths.metadata))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leak-free classification train/val/test splits."
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    paths = SplitPaths(
        pool=resolve_repo_path(args.pool),
        golden=resolve_repo_path(args.golden),
        rag=resolve_repo_path(args.rag),
        output_dir=resolve_repo_path(args.out_dir),
        metadata=resolve_repo_path(args.metadata),
    )
    try:
        build_splits(paths, seed=args.seed)
    except SplitError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
