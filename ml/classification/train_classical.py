from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classification.data import (
    LABEL_ORDER,
    ClassificationSplitPaths,
    display_path,
    load_classification_splits,
    resolve_repo_path,
)
from classification.evaluate_classical import DEFAULT_ARTIFACT_DIR, evaluate_model, write_json

JsonObject = dict[str, Any]

RANDOM_SEED = 42
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "model.joblib"
DEFAULT_VAL_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "val_metrics.json"
DEFAULT_LABEL_MAP_PATH = DEFAULT_ARTIFACT_DIR / "label_map.json"
DEFAULT_TRAINING_SUMMARY_PATH = DEFAULT_ARTIFACT_DIR / "training_summary.json"


def build_pipeline(*, seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=50_000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def label_map() -> JsonObject:
    return {
        "id_to_label": {str(index): label for index, label in enumerate(LABEL_ORDER)},
        "label_to_id": {label: index for index, label in enumerate(LABEL_ORDER)},
    }


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def train_classical(
    *,
    split_paths: ClassificationSplitPaths | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    val_metrics_path: Path = DEFAULT_VAL_METRICS_PATH,
    label_map_path: Path = DEFAULT_LABEL_MAP_PATH,
    training_summary_path: Path = DEFAULT_TRAINING_SUMMARY_PATH,
    seed: int = RANDOM_SEED,
) -> JsonObject:
    if split_paths is None:
        split_paths = ClassificationSplitPaths()
    splits = load_classification_splits(split_paths)
    model = build_pipeline(seed=seed)
    model.fit(splits.train.texts, splits.train.labels)

    resolved_model_path = resolve_repo_path(model_path)
    save_model(model, resolved_model_path)

    val_metrics = evaluate_model(model, splits.val, model_path=resolved_model_path)
    write_json(val_metrics, resolve_repo_path(val_metrics_path))
    write_json(label_map(), resolve_repo_path(label_map_path))

    summary: JsonObject = {
        "artifact_dir": display_path(resolve_repo_path(model_path).parent),
        "label_map_path": display_path(resolve_repo_path(label_map_path)),
        "model_path": display_path(resolved_model_path),
        "random_seed": seed,
        "train_examples": len(splits.train),
        "training_algorithm": "tfidf_logistic_regression",
        "val_examples": len(splits.val),
        "val_metrics_path": display_path(resolve_repo_path(val_metrics_path)),
    }
    write_json(summary, resolve_repo_path(training_summary_path))
    return {"summary": summary, "val_metrics": val_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classical classification baseline.")
    parser.add_argument("--train-path", type=Path, default=ClassificationSplitPaths.train)
    parser.add_argument("--val-path", type=Path, default=ClassificationSplitPaths.val)
    parser.add_argument("--test-path", type=Path, default=ClassificationSplitPaths.test)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--val-metrics-path", type=Path, default=DEFAULT_VAL_METRICS_PATH)
    parser.add_argument("--label-map-path", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--training-summary-path", type=Path, default=DEFAULT_TRAINING_SUMMARY_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_paths = ClassificationSplitPaths(
        train=resolve_repo_path(args.train_path),
        val=resolve_repo_path(args.val_path),
        test=resolve_repo_path(args.test_path),
    )
    result = train_classical(
        split_paths=split_paths,
        model_path=args.model_path,
        val_metrics_path=args.val_metrics_path,
        label_map_path=args.label_map_path,
        training_summary_path=args.training_summary_path,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
