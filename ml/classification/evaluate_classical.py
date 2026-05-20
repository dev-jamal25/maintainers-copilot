from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from classification.data import (
    LABEL_ORDER,
    REPO_ROOT,
    ClassificationDataset,
    ClassificationSplitPaths,
    display_path,
    load_classification_dataset,
    resolve_repo_path,
)

JsonObject = dict[str, Any]

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "classification" / "classical"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "test_metrics.json"


class TextClassifier(Protocol):
    def predict(self, texts: list[str]) -> Any: ...


def write_json(record: JsonObject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def model_size_bytes(path: Path) -> int:
    return path.stat().st_size


def evaluate_model(
    model: TextClassifier,
    dataset: ClassificationDataset,
    *,
    model_path: Path | None = None,
) -> JsonObject:
    started = time.perf_counter()
    predictions = [str(label) for label in model.predict(dataset.texts)]
    latency_seconds = time.perf_counter() - started
    labels = list(LABEL_ORDER)
    truth = dataset.labels
    per_class_f1_values = f1_score(
        truth,
        predictions,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    matrix = confusion_matrix(truth, predictions, labels=labels)
    model_size = (
        model_size_bytes(model_path) if model_path is not None and model_path.exists() else None
    )

    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        "latency": {
            "examples": len(dataset),
            "ms_per_example": (latency_seconds / len(dataset)) * 1000,
            "total_seconds": latency_seconds,
        },
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0.0)
        ),
        "model": {
            "path": display_path(model_path) if model_path is not None else None,
            "size_bytes": model_size,
        },
        "per_class_f1": {
            label: float(score) for label, score in zip(labels, per_class_f1_values, strict=True)
        },
        "split": dataset.split,
    }


def load_model(path: Path) -> TextClassifier:
    loaded = joblib.load(path)
    return cast(TextClassifier, loaded)


def dataset_for_split(split: str, paths: ClassificationSplitPaths) -> ClassificationDataset:
    split_paths = {
        "train": paths.train,
        "val": paths.val,
        "test": paths.test,
    }
    if split not in split_paths:
        raise ValueError(f"split must be one of {sorted(split_paths)}")
    return load_classification_dataset(split_paths[split], split=split)


def evaluate_classical(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    split: str = "test",
    split_paths: ClassificationSplitPaths | None = None,
) -> JsonObject:
    if split_paths is None:
        split_paths = ClassificationSplitPaths()
    resolved_model_path = resolve_repo_path(model_path)
    model = load_model(resolved_model_path)
    dataset = dataset_for_split(split, split_paths)
    metrics = evaluate_model(model, dataset, model_path=resolved_model_path)
    write_json(metrics, resolve_repo_path(metrics_path))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the classical classification baseline.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--train-path", type=Path, default=ClassificationSplitPaths.train)
    parser.add_argument("--val-path", type=Path, default=ClassificationSplitPaths.val)
    parser.add_argument("--test-path", type=Path, default=ClassificationSplitPaths.test)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_paths = ClassificationSplitPaths(
        train=resolve_repo_path(args.train_path),
        val=resolve_repo_path(args.val_path),
        test=resolve_repo_path(args.test_path),
    )
    metrics = evaluate_classical(
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        split=args.split,
        split_paths=split_paths,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
