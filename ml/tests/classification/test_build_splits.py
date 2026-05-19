from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from classification.build_splits import SplitError, SplitPaths, build_splits

JsonObject = dict[str, Any]

LABEL_IDS = {"bug": 0, "feature": 1, "docs": 2, "question": 3}
LABELS = ("bug", "feature", "docs", "question")


def _timestamp(offset_days: int) -> str:
    value = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=offset_days)
    return value.isoformat().replace("+00:00", "Z")


def _record(index: int, *, label: str | None = None, created_at: str | None = None) -> JsonObject:
    resolved_label = label or LABELS[index % len(LABELS)]
    return {
        "closed_at": _timestamp(index + 1),
        "created_at": created_at or _timestamp(index),
        "fetched_for_github_label": f"kind:{resolved_label}",
        "fetched_for_label": resolved_label,
        "github_id": 10_000 + index,
        "html_url": f"https://github.com/apache/airflow/issues/{index}",
        "label": resolved_label,
        "label_id": LABEL_IDS.get(resolved_label, 99),
        "number": index,
        "text": f"Issue {index} body for {resolved_label}",
    }


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_paths(
    tmp_path: Path,
    *,
    pool: list[JsonObject],
    golden: list[JsonObject] | None = None,
    rag: list[JsonObject] | None = None,
    metadata: JsonObject | None = None,
) -> SplitPaths:
    split_dir = tmp_path / "data" / "splits"
    out_dir = tmp_path / "data" / "processed"
    metadata_path = out_dir / "dataset_metadata.json"
    pool_path = split_dir / "splittable_pool.jsonl"
    golden_path = split_dir / "golden_eval.jsonl"
    rag_path = split_dir / "rag_holdout.jsonl"

    _write_jsonl(pool_path, pool)
    _write_jsonl(golden_path, golden or [])
    _write_jsonl(rag_path, rag or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata or {"existing": "preserve"}), encoding="utf-8")

    return SplitPaths(
        pool=pool_path,
        golden=golden_path,
        rag=rag_path,
        output_dir=out_dir,
        metadata=metadata_path,
    )


def _pool(count: int = 120) -> list[JsonObject]:
    return [_record(index) for index in range(count)]


def _assert_all_labels_present(records_by_split: dict[str, list[JsonObject]]) -> None:
    for split_name, records in records_by_split.items():
        labels = {str(record["label"]) for record in records}
        assert labels == set(LABELS), split_name


def _assert_train_before_test_per_label(train: list[JsonObject], test: list[JsonObject]) -> None:
    for label in LABELS:
        label_train = [record for record in train if record["label"] == label]
        label_test = [record for record in test if record["label"] == label]
        train_max = max(str(record["created_at"]) for record in label_train)
        test_min = min(str(record["created_at"]) for record in label_test)
        assert train_max < test_min, label


def test_overlap_with_reserve_files_raises_clear_error(tmp_path: Path) -> None:
    pool = _pool()
    reserved = [{"github_id": pool[0]["github_id"], "number": pool[0]["number"]}]
    paths = _make_paths(tmp_path, pool=pool, golden=reserved)

    with pytest.raises(SplitError, match=str(pool[0]["github_id"])):
        build_splits(paths)


def test_output_split_files_are_created(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, pool=_pool())

    build_splits(paths)

    assert (paths.output_dir / "classification_train.jsonl").is_file()
    assert (paths.output_dir / "classification_val.jsonl").is_file()
    assert (paths.output_dir / "classification_test.jsonl").is_file()
    assert _read_jsonl(paths.output_dir / "classification_train.jsonl")
    assert _read_jsonl(paths.output_dir / "classification_val.jsonl")
    assert _read_jsonl(paths.output_dir / "classification_test.jsonl")


def test_each_label_appears_in_train_val_and_test(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, pool=_pool())

    build_splits(paths)

    train = _read_jsonl(paths.output_dir / "classification_train.jsonl")
    val = _read_jsonl(paths.output_dir / "classification_val.jsonl")
    test = _read_jsonl(paths.output_dir / "classification_test.jsonl")
    _assert_all_labels_present({"train": train, "val": val, "test": test})


def test_per_label_test_timestamps_are_newer_than_train_timestamps(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, pool=_pool())

    build_splits(paths)

    train = _read_jsonl(paths.output_dir / "classification_train.jsonl")
    test = _read_jsonl(paths.output_dir / "classification_test.jsonl")
    _assert_train_before_test_per_label(train, test)


def test_invalid_labels_raise_clear_error(tmp_path: Path) -> None:
    pool = _pool()
    pool[0]["label"] = "chore"
    paths = _make_paths(tmp_path, pool=pool)

    with pytest.raises(SplitError, match="invalid label 'chore'"):
        build_splits(paths)


def test_missing_required_fields_raise_clear_error(tmp_path: Path) -> None:
    pool = _pool()
    del pool[0]["created_at"]
    paths = _make_paths(tmp_path, pool=pool)

    with pytest.raises(SplitError, match="missing required field\\(s\\): created_at"):
        build_splits(paths)


def test_metadata_is_updated_and_existing_keys_are_preserved(tmp_path: Path) -> None:
    paths = _make_paths(
        tmp_path,
        pool=_pool(),
        metadata={"preserved_top_level_key": {"keep": True}},
    )

    summary = build_splits(paths)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    block = metadata["classification_splits"]

    assert metadata["preserved_top_level_key"] == {"keep": True}
    assert block["input_pool"]["count"] == summary.input_count
    assert block["splits"]["train"]["sha256"] == summary.output_hashes["train"]
    assert block["splits"]["val"]["per_class"]
    assert block["split_policy"]["temporal_policy"] == "per_class_created_at"
    assert block["splits"]["train"]["per_class_created_at"]["bug"]["max_created_at"]
    assert block["splits"]["test"]["min_created_at"]


def test_split_generation_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, pool=_pool())

    first = build_splits(paths)
    second = build_splits(paths)

    assert second.output_hashes == first.output_hashes
    assert second.split_counts == first.split_counts


def test_timestamp_ties_at_boundary_preserve_temporal_invariant(tmp_path: Path) -> None:
    pool: list[JsonObject] = []
    for label_index, label in enumerate(LABELS):
        for label_offset in range(20):
            created_at = _timestamp(label_offset)
            if label == "bug" and 14 <= label_offset <= 17:
                created_at = _timestamp(14)
            pool.append(
                _record(label_index * 100 + label_offset, label=label, created_at=created_at)
            )
    paths = _make_paths(tmp_path, pool=pool)

    summary = build_splits(paths)
    train = _read_jsonl(paths.output_dir / "classification_train.jsonl")
    test = _read_jsonl(paths.output_dir / "classification_test.jsonl")

    _assert_train_before_test_per_label(train, test)
    assert summary.per_class_counts["test"]["bug"] == 6
