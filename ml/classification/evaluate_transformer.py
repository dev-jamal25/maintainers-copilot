from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from classification.data import (
    ClassificationSplitPaths,
    display_path,
    load_classification_dataset,
    resolve_repo_path,
)
from classification.train_transformer import (
    TransformerTextDataset,
    build_data_collator,
    model_artifact_path,
    prediction_metrics,
    selected_device_info,
    sha256_file,
    tokenize_dataset,
    trainer_kwargs,
    write_json,
)
from classification.training_config import TrainingConfig

JsonObject = dict[str, Any]
DEFAULT_MODEL_DIR = TrainingConfig().resolved_output_dir
DEFAULT_EVAL_METRICS_PATH = TrainingConfig().resolved_output_dir / "test_metrics.json"


def dataset_path_for_split(split: str, paths: ClassificationSplitPaths) -> Path:
    split_paths = {
        "train": paths.train,
        "val": paths.val,
        "test": paths.test,
    }
    if split not in split_paths:
        raise ValueError(f"split must be one of {sorted(split_paths)}")
    return split_paths[split]


def max_length_for_model_dir(model_dir: Path, explicit_max_length: int | None = None) -> int:
    if explicit_max_length is not None:
        return explicit_max_length

    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        return TrainingConfig().max_length

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    value = config.get("max_length")
    if not isinstance(value, int):
        return TrainingConfig().max_length
    return value


def evaluate_transformer_dataset(
    *,
    model: Any,
    tokenizer: Any,
    dataset: TransformerTextDataset,
    split: str,
    output_dir: Path,
    model_dir: Path,
) -> JsonObject:
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=TrainingConfig().batch_size,
        report_to="none",
    )
    trainer = Trainer(
        args=args,
        data_collator=build_data_collator(tokenizer),
        model=model,
        **trainer_kwargs(tokenizer=tokenizer),
    )
    started = time.perf_counter()
    prediction = trainer.predict(dataset, metric_key_prefix=split)
    latency_seconds = time.perf_counter() - started
    metrics = prediction_metrics(
        np.asarray(prediction.label_ids),
        np.asarray(prediction.predictions),
        split=split,
    )
    artifact_path = model_artifact_path(model_dir)
    metrics["latency"] = {
        "examples": len(dataset),
        "ms_per_example": (latency_seconds / len(dataset)) * 1000,
        "total_seconds": latency_seconds,
    }
    metrics["device"] = selected_device_info()
    metrics["model"] = {
        "artifact_path": display_path(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
    }
    return metrics


def evaluate_transformer(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_path: Path = DEFAULT_EVAL_METRICS_PATH,
    split: str = "test",
    split_paths: ClassificationSplitPaths | None = None,
    max_length: int | None = None,
) -> JsonObject:
    if split_paths is None:
        split_paths = ClassificationSplitPaths()
    resolved_model_dir = resolve_repo_path(model_dir)
    dataset = load_classification_dataset(dataset_path_for_split(split, split_paths), split=split)
    resolved_max_length = max_length_for_model_dir(resolved_model_dir, max_length)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model_dir)
    tokenized = tokenize_dataset(tokenizer, dataset, max_length=resolved_max_length)
    metrics = evaluate_transformer_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized,
        split=split,
        output_dir=resolved_model_dir / "eval",
        model_dir=resolved_model_dir,
    )
    write_json(metrics, resolve_repo_path(metrics_path))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DistilBERT classifier.")
    parser.add_argument("--model-dir", type=Path, default=TrainingConfig().output_dir)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_EVAL_METRICS_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--train-path", type=Path, default=ClassificationSplitPaths.train)
    parser.add_argument("--val-path", type=Path, default=ClassificationSplitPaths.val)
    parser.add_argument("--test-path", type=Path, default=ClassificationSplitPaths.test)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_paths = ClassificationSplitPaths(
        train=resolve_repo_path(args.train_path),
        val=resolve_repo_path(args.val_path),
        test=resolve_repo_path(args.test_path),
    )
    metrics = evaluate_transformer(
        model_dir=args.model_dir,
        metrics_path=args.metrics_path,
        split=args.split,
        split_paths=split_paths,
        max_length=args.max_length,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
