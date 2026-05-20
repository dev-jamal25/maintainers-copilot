from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from classification.data import ClassificationSplitPaths, label_to_id
from classification.evaluate_classical import evaluate_classical
from classification.train_classical import train_classical

JsonObject = dict[str, Any]

LABEL_TERMS = {
    "bug": "crash failure traceback exception",
    "feature": "feature support request enhancement",
    "docs": "documentation guide example readme",
    "question": "question help how configure",
}


def _timestamp(offset_days: int) -> str:
    value = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=offset_days)
    return value.isoformat().replace("+00:00", "Z")


def _record(index: int, *, label: str) -> JsonObject:
    return {
        "created_at": _timestamp(index),
        "github_id": 100_000 + index,
        "html_url": f"https://github.com/apache/airflow/issues/{index}",
        "label": label,
        "label_id": label_to_id(label),
        "number": index,
        "text": f"{LABEL_TERMS[label]} airflow issue {index}",
    }


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _write_splits(tmp_path: Path) -> ClassificationSplitPaths:
    train_records: list[JsonObject] = []
    val_records: list[JsonObject] = []
    test_records: list[JsonObject] = []
    index = 0
    for label in LABEL_TERMS:
        for _ in range(8):
            train_records.append(_record(index, label=label))
            index += 1
        for _ in range(2):
            val_records.append(_record(index, label=label))
            index += 1
        for _ in range(2):
            test_records.append(_record(index, label=label))
            index += 1

    paths = ClassificationSplitPaths(
        train=tmp_path / "classification_train.jsonl",
        val=tmp_path / "classification_val.jsonl",
        test=tmp_path / "classification_test.jsonl",
    )
    _write_jsonl(paths.train, train_records)
    _write_jsonl(paths.val, val_records)
    _write_jsonl(paths.test, test_records)
    return paths


def test_train_classical_writes_model_metrics_and_label_map(tmp_path: Path) -> None:
    split_paths = _write_splits(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    result = train_classical(
        split_paths=split_paths,
        model_path=artifact_dir / "model.joblib",
        val_metrics_path=artifact_dir / "val_metrics.json",
        label_map_path=artifact_dir / "label_map.json",
        training_summary_path=artifact_dir / "training_summary.json",
    )

    assert (artifact_dir / "model.joblib").is_file()
    assert (artifact_dir / "val_metrics.json").is_file()
    assert (artifact_dir / "label_map.json").is_file()
    assert (artifact_dir / "training_summary.json").is_file()
    assert result["val_metrics"]["split"] == "val"
    assert set(result["val_metrics"]["per_class_f1"]) == set(LABEL_TERMS)
    assert result["val_metrics"]["model"]["size_bytes"] > 0


def test_evaluate_classical_writes_required_metrics(tmp_path: Path) -> None:
    split_paths = _write_splits(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    model_path = artifact_dir / "model.joblib"
    train_classical(split_paths=split_paths, model_path=model_path)

    metrics = evaluate_classical(
        model_path=model_path,
        metrics_path=artifact_dir / "test_metrics.json",
        split="test",
        split_paths=split_paths,
    )

    assert (artifact_dir / "test_metrics.json").is_file()
    assert metrics["split"] == "test"
    assert metrics["accuracy"] >= 0.0
    assert metrics["macro_f1"] >= 0.0
    assert metrics["confusion_matrix"]["labels"] == ["bug", "feature", "docs", "question"]
    assert len(metrics["confusion_matrix"]["matrix"]) == 4
    assert metrics["latency"]["examples"] == 8
    assert metrics["model"]["size_bytes"] > 0
