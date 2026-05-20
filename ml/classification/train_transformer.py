from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from classification.data import (
    ID_TO_LABEL,
    LABEL_ORDER,
    LABEL_TO_ID,
    ClassificationDataset,
    ClassificationSplitPaths,
    display_path,
    load_classification_splits,
    resolve_repo_path,
)
from classification.training_config import FreezePolicy, TrainingConfig, default_training_config

JsonObject = dict[str, Any]

DEFAULT_METRICS_PATH = TrainingConfig().resolved_output_dir / "metrics.json"
DEFAULT_LABEL_MAP_PATH = TrainingConfig().resolved_output_dir / "label_map.json"
DEFAULT_MODEL_CARD_PATH = TrainingConfig().resolved_output_dir / "model_card.md"
DEFAULT_CONFIG_PATH = TrainingConfig().resolved_output_dir / "training_config.json"
METRIC_LIBRARY = "evaluate"
USES_DYNAMIC_PADDING = True
_ACCURACY_METRIC: Any | None = None
_F1_METRIC: Any | None = None


class TransformerTextDataset(Dataset[dict[str, list[int] | int]]):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        item: dict[str, list[int] | int] = {
            key: values[index] for key, values in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


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


def label_map() -> JsonObject:
    return {
        "id_to_label": {str(index): label for index, label in ID_TO_LABEL.items()},
        "label_to_id": dict(LABEL_TO_ID),
    }


def selected_device_info() -> JsonObject:
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        return {
            "cuda_available": True,
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(device_index),
            "torch_cuda_version": torch.version.cuda,
        }
    return {
        "cuda_available": False,
        "device": "cpu",
        "device_name": "CPU only",
        "torch_cuda_version": torch.version.cuda,
    }


def pin_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def tokenize_dataset(
    tokenizer: Any,
    dataset: ClassificationDataset,
    *,
    max_length: int,
) -> TransformerTextDataset:
    encodings = tokenizer(
        dataset.texts,
        max_length=max_length,
        truncation=True,
    )
    return TransformerTextDataset(dict(encodings), dataset.label_ids)


def build_data_collator(tokenizer: Any) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def load_sequence_classification_model(config: TrainingConfig) -> Any:
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        num_labels=len(LABEL_ORDER),
    )


def evaluate_metric_modules() -> tuple[Any, Any]:
    global _ACCURACY_METRIC, _F1_METRIC
    if _ACCURACY_METRIC is None:
        _ACCURACY_METRIC = evaluate.load("accuracy")
    if _F1_METRIC is None:
        _F1_METRIC = evaluate.load("f1")
    return _ACCURACY_METRIC, _F1_METRIC


def prediction_metrics(label_ids: np.ndarray, predictions: np.ndarray, *, split: str) -> JsonObject:
    predicted_ids = predictions.argmax(axis=-1)
    labels = list(range(len(LABEL_ORDER)))
    per_class_f1 = f1_score(
        label_ids,
        predicted_ids,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    matrix = confusion_matrix(label_ids, predicted_ids, labels=labels)
    return {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "confusion_matrix": {
            "labels": list(LABEL_ORDER),
            "matrix": matrix.tolist(),
        },
        "macro_f1": float(
            f1_score(label_ids, predicted_ids, labels=labels, average="macro", zero_division=0.0)
        ),
        "per_class_f1": {
            label: float(score) for label, score in zip(LABEL_ORDER, per_class_f1, strict=True)
        },
        "split": split,
    }


def trainer_compute_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    references = np.asarray(eval_prediction.label_ids)
    predicted_ids = np.asarray(predictions).argmax(axis=-1)
    accuracy_metric, f1_metric = evaluate_metric_modules()
    accuracy = accuracy_metric.compute(predictions=predicted_ids, references=references)
    macro_f1 = f1_metric.compute(
        predictions=predicted_ids,
        references=references,
        average="macro",
    )
    return {
        "accuracy": float(accuracy["accuracy"]),
        "macro_f1": float(macro_f1["f1"]),
    }


def apply_freeze_policy(model: Any, policy: FreezePolicy) -> None:
    if not policy.freeze_embeddings and not policy.freeze_encoder_layer_indices:
        return

    distilbert = getattr(model, "distilbert", None)
    if distilbert is None:
        raise ValueError("expected model to expose a distilbert attribute")

    if policy.freeze_embeddings:
        embeddings = distilbert.embeddings
        for parameter in embeddings.parameters():
            parameter.requires_grad = False

    transformer = distilbert.transformer
    layers = transformer.layer
    for layer_index in policy.freeze_encoder_layer_indices:
        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = False


def model_artifact_path(output_dir: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = output_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"no model artifact found in {display_path(output_dir)}")


def training_arguments_kwargs(
    config: TrainingConfig,
    *,
    enable_wandb: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_eval": True,
        "greater_is_better": True,
        "learning_rate": config.learning_rate,
        "load_best_model_at_end": True,
        "logging_strategy": "epoch",
        "metric_for_best_model": "macro_f1",
        "num_train_epochs": config.epochs,
        "output_dir": str(config.resolved_output_dir),
        "per_device_eval_batch_size": config.batch_size,
        "per_device_train_batch_size": config.batch_size,
        "report_to": ["wandb"] if enable_wandb else "none",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "seed": config.seed,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def trainer_kwargs(*, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def should_enable_wandb(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return bool(os.getenv("WANDB_API_KEY") or os.getenv("WANDB_MODE"))


def write_model_card(
    *,
    config: TrainingConfig,
    metrics: JsonObject,
    model_hash: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    val_macro_f1 = metrics["val"]["macro_f1"] if isinstance(metrics.get("val"), dict) else "n/a"
    test_macro_f1 = metrics["test"]["macro_f1"] if isinstance(metrics.get("test"), dict) else "n/a"
    content = f"""# DistilBERT Classification Model

Base model: `{config.model_name}`

Device: `{metrics.get("device", {}).get("device_name", "unknown")}`

CUDA available: `{metrics.get("device", {}).get("cuda_available", "unknown")}`

Seed: `{config.seed}`

Dynamic padding: `{USES_DYNAMIC_PADDING}`

Trainer metric library: `{METRIC_LIBRARY}`

Task: classify apache/airflow GitHub issues into `{", ".join(LABEL_ORDER)}`.

Training data: Phase 1B per-class temporal train/val/test splits.

Validation macro-F1: `{val_macro_f1}`

Test macro-F1: `{test_macro_f1}`

Model artifact SHA-256: `{model_hash}`

Config:

```json
{json.dumps(config.to_dict(), indent=2, sort_keys=True)}
```
"""
    path.write_text(content, encoding="utf-8")


def train_transformer(
    *,
    config: TrainingConfig | None = None,
    split_paths: ClassificationSplitPaths | None = None,
    enable_wandb: bool | None = None,
) -> JsonObject:
    if config is None:
        config = default_training_config()
    if split_paths is None:
        split_paths = ClassificationSplitPaths()

    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    resolved_wandb = should_enable_wandb(enable_wandb)
    pin_all_seeds(config.seed)

    splits = load_classification_splits(split_paths)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_sequence_classification_model(config)
    apply_freeze_policy(model, config.freeze_policy)

    train_dataset = tokenize_dataset(tokenizer, splits.train, max_length=config.max_length)
    val_dataset = tokenize_dataset(tokenizer, splits.val, max_length=config.max_length)
    test_dataset = tokenize_dataset(tokenizer, splits.test, max_length=config.max_length)
    data_collator = build_data_collator(tokenizer)

    args = TrainingArguments(**training_arguments_kwargs(config, enable_wandb=resolved_wandb))
    trainer = Trainer(
        args=args,
        compute_metrics=trainer_compute_metrics,
        data_collator=data_collator,
        eval_dataset=val_dataset,
        model=model,
        train_dataset=train_dataset,
        **trainer_kwargs(tokenizer=tokenizer),
    )

    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started
    trainer.save_model(str(config.resolved_output_dir))
    tokenizer.save_pretrained(config.resolved_output_dir)

    val_prediction = trainer.predict(val_dataset, metric_key_prefix="val")
    test_prediction = trainer.predict(test_dataset, metric_key_prefix="test")
    artifact_path = model_artifact_path(config.resolved_output_dir)
    model_hash = sha256_file(artifact_path)

    metrics: JsonObject = {
        "device": selected_device_info(),
        "model": {
            "artifact_path": display_path(artifact_path),
            "artifact_sha256": model_hash,
            "output_dir": display_path(config.resolved_output_dir),
        },
        "test": prediction_metrics(
            np.asarray(test_prediction.label_ids),
            np.asarray(test_prediction.predictions),
            split="test",
        ),
        "train": {
            "examples": len(splits.train),
            "metrics": train_result.metrics,
            "seconds": training_seconds,
        },
        "training_setup": {
            "dynamic_padding": USES_DYNAMIC_PADDING,
            "max_length": config.max_length,
            "metric_library": METRIC_LIBRARY,
            "seed": config.seed,
        },
        "val": prediction_metrics(
            np.asarray(val_prediction.label_ids),
            np.asarray(val_prediction.predictions),
            split="val",
        ),
        "wandb": {
            "enabled": resolved_wandb,
            "project": config.wandb_project,
        },
    }
    metrics_path = config.resolved_output_dir / "metrics.json"
    label_map_path = config.resolved_output_dir / "label_map.json"
    config_path = config.resolved_output_dir / "training_config.json"
    model_card_path = config.resolved_output_dir / "model_card.md"
    write_json(metrics, metrics_path)
    write_json(label_map(), label_map_path)
    write_json(config.to_dict(), config_path)
    write_model_card(
        config=config,
        metrics=metrics,
        model_hash=model_hash,
        path=model_card_path,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for issue classification.")
    parser.add_argument("--train-path", type=Path, default=ClassificationSplitPaths.train)
    parser.add_argument("--val-path", type=Path, default=ClassificationSplitPaths.val)
    parser.add_argument("--test-path", type=Path, default=ClassificationSplitPaths.test)
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig().output_dir)
    parser.add_argument("--max-length", type=int, default=TrainingConfig().max_length)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig().learning_rate)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=resolve_repo_path(args.output_dir),
        seed=args.seed,
    )
    split_paths = ClassificationSplitPaths(
        train=resolve_repo_path(args.train_path),
        val=resolve_repo_path(args.val_path),
        test=resolve_repo_path(args.test_path),
    )
    metrics = train_transformer(config=config, split_paths=split_paths, enable_wandb=args.wandb)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
