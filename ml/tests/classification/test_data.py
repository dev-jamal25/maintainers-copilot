from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from classification.data import (
    LABEL_ORDER,
    ClassificationDataError,
    ClassificationSplitPaths,
    id_to_label,
    label_to_id,
    load_classification_dataset,
    load_classification_splits,
    normalize_text,
)

JsonObject = dict[str, Any]


def _timestamp(offset_days: int = 0) -> str:
    value = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=offset_days)
    return value.isoformat().replace("+00:00", "Z")


def _record(
    index: int,
    *,
    label: str = "bug",
    label_id: int | None = None,
    text: str | None = None,
) -> JsonObject:
    return {
        "created_at": _timestamp(index),
        "github_id": 50_000 + index,
        "html_url": f"https://github.com/apache/airflow/issues/{index}",
        "label": label,
        "label_id": label_to_id(label) if label_id is None else label_id,
        "number": index,
        "text": text if text is not None else f"Issue {index}\n\nBody",
    }


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def test_label_mapping_is_stable() -> None:
    assert LABEL_ORDER == ("bug", "feature", "docs", "question")
    assert label_to_id("bug") == 0
    assert label_to_id("feature") == 1
    assert label_to_id("docs") == 2
    assert label_to_id("question") == 3
    assert id_to_label(0) == "bug"
    assert id_to_label(3) == "question"


def test_unknown_label_mapping_raises_clear_error() -> None:
    with pytest.raises(ClassificationDataError, match="unknown label 'chore'"):
        label_to_id("chore")
    with pytest.raises(ClassificationDataError, match="unknown label id 99"):
        id_to_label(99)


def test_normalize_text_trims_line_endings_and_spacing() -> None:
    assert normalize_text("  hello\t world\r\n\r\n\r\nnext  line  ") == "hello world\n\nnext line"


def test_load_classification_dataset_validates_and_normalizes_records(tmp_path: Path) -> None:
    path = tmp_path / "classification_train.jsonl"
    _write_jsonl(
        path,
        [
            _record(1, label="bug", text="  Title\r\n\r\n\r\nBody\t text  "),
            _record(2, label="docs"),
        ],
    )

    dataset = load_classification_dataset(path, split="train")

    assert dataset.split == "train"
    assert len(dataset) == 2
    assert dataset[0].issue_id == "50001"
    assert dataset[0].text == "Title\n\nBody text"
    assert dataset.texts == ["Title\n\nBody text", "Issue 2\n\nBody"]
    assert dataset.labels == ["bug", "docs"]
    assert dataset.label_ids == [0, 2]


def test_load_classification_splits_loads_train_val_and_test(tmp_path: Path) -> None:
    train_path = tmp_path / "classification_train.jsonl"
    val_path = tmp_path / "classification_val.jsonl"
    test_path = tmp_path / "classification_test.jsonl"
    _write_jsonl(train_path, [_record(1, label="bug")])
    _write_jsonl(val_path, [_record(2, label="feature")])
    _write_jsonl(test_path, [_record(3, label="question")])

    splits = load_classification_splits(
        ClassificationSplitPaths(train=train_path, val=val_path, test=test_path)
    )

    assert splits.train.labels == ["bug"]
    assert splits.val.labels == ["feature"]
    assert splits.test.labels == ["question"]


def test_missing_required_field_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "classification_train.jsonl"
    record = _record(1)
    del record["text"]
    _write_jsonl(path, [record])

    with pytest.raises(ClassificationDataError, match="missing required field 'text'"):
        load_classification_dataset(path, split="train")


def test_label_id_mismatch_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "classification_train.jsonl"
    _write_jsonl(path, [_record(1, label="docs", label_id=0)])

    with pytest.raises(ClassificationDataError, match="label_id 0 does not match label 'docs'"):
        load_classification_dataset(path, split="train")


def test_empty_normalized_text_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "classification_train.jsonl"
    _write_jsonl(path, [_record(1, text=" \r\n\t ")])

    with pytest.raises(ClassificationDataError, match="field 'text' is empty"):
        load_classification_dataset(path, split="train")
