from __future__ import annotations

import argparse
import json
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
    ID_TO_LABEL,
    LABEL_ORDER,
    ClassificationDataset,
    ClassificationSplitPaths,
    display_path,
    load_classification_dataset,
    resolve_repo_path,
)
from classification.evaluate_transformer import dataset_path_for_split, max_length_for_model_dir
from classification.train_transformer import (
    build_data_collator,
    prediction_metrics,
    selected_device_info,
    tokenize_dataset,
    trainer_kwargs,
    write_json,
)
from classification.training_config import TrainingConfig

JsonObject = dict[str, Any]
DEFAULT_MODEL_DIR = TrainingConfig().resolved_output_dir
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "error_analysis"
PREVIEW_LIMIT = 500


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=1, keepdims=True))


def text_preview(text: str, *, limit: int = PREVIEW_LIMIT) -> str:
    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3].rstrip() + "..."


def prediction_error_rows(
    *,
    dataset: ClassificationDataset,
    logits: np.ndarray,
    split: str,
) -> list[JsonObject]:
    probabilities = softmax(logits)
    predicted_ids = probabilities.argmax(axis=1)
    rows: list[JsonObject] = []
    for example, predicted_id, label_probabilities in zip(
        dataset,
        predicted_ids.tolist(),
        probabilities.tolist(),
        strict=True,
    ):
        if predicted_id == example.label_id:
            continue
        predicted_label = ID_TO_LABEL[predicted_id]
        rows.append(
            {
                "confidence": float(label_probabilities[predicted_id]),
                "created_at": example.created_at.isoformat(),
                "html_url": example.html_url,
                "issue_id": example.issue_id,
                "number": example.number,
                "predicted_label": predicted_label,
                "probabilities": {
                    label: float(label_probabilities[index])
                    for index, label in enumerate(LABEL_ORDER)
                },
                "split": split,
                "text_preview": text_preview(example.text),
                "true_label": example.label,
            }
        )
    return rows


def per_class_error_summary(rows: list[JsonObject], dataset: ClassificationDataset) -> JsonObject:
    totals = {label: 0 for label in LABEL_ORDER}
    errors = {label: 0 for label in LABEL_ORDER}
    predicted_as = {label: {predicted: 0 for predicted in LABEL_ORDER} for label in LABEL_ORDER}
    for example in dataset:
        totals[example.label] += 1
    for row in rows:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])
        errors[true_label] += 1
        predicted_as[true_label][predicted_label] += 1
    return {
        label: {
            "error_count": errors[label],
            "error_rate": errors[label] / totals[label] if totals[label] else 0.0,
            "predicted_as": predicted_as[label],
            "total": totals[label],
        }
        for label in LABEL_ORDER
    }


def write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def predict_logits(
    *,
    model: Any,
    tokenizer: Any,
    dataset: ClassificationDataset,
    split: str,
    output_dir: Path,
    max_length: int,
) -> np.ndarray:
    tokenized = tokenize_dataset(tokenizer, dataset, max_length=max_length)
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
    prediction = trainer.predict(tokenized, metric_key_prefix=split)
    return np.asarray(prediction.predictions)


def analyze_split(
    *,
    model: Any,
    tokenizer: Any,
    dataset: ClassificationDataset,
    split: str,
    output_dir: Path,
    max_length: int,
) -> tuple[list[JsonObject], JsonObject, JsonObject]:
    logits = predict_logits(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        split=split,
        output_dir=output_dir / f"{split}_eval",
        max_length=max_length,
    )
    rows = prediction_error_rows(dataset=dataset, logits=logits, split=split)
    metrics = prediction_metrics(np.asarray(dataset.label_ids), logits, split=split)
    summary = per_class_error_summary(rows, dataset)
    return rows, metrics, summary


def question_as_bug(rows: list[JsonObject]) -> list[JsonObject]:
    return [
        row for row in rows if row["true_label"] == "question" and row["predicted_label"] == "bug"
    ]


def analyze_transformer_errors(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_paths: ClassificationSplitPaths | None = None,
    max_length: int | None = None,
) -> JsonObject:
    if split_paths is None:
        split_paths = ClassificationSplitPaths()
    resolved_model_dir = resolve_repo_path(model_dir)
    resolved_output_dir = resolve_repo_path(output_dir)
    resolved_max_length = max_length_for_model_dir(resolved_model_dir, max_length)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model_dir)
    datasets = {
        split: load_classification_dataset(dataset_path_for_split(split, split_paths), split=split)
        for split in ("val", "test")
    }

    split_errors: dict[str, list[JsonObject]] = {}
    confusion_summary: JsonObject = {
        "device": selected_device_info(),
        "max_length": resolved_max_length,
        "model_dir": display_path(resolved_model_dir),
    }
    error_summary: JsonObject = {}
    for split, dataset in datasets.items():
        rows, metrics, summary = analyze_split(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            split=split,
            output_dir=resolved_output_dir,
            max_length=resolved_max_length,
        )
        split_errors[split] = rows
        confusion_summary[split] = metrics
        error_summary[split] = summary
        write_jsonl(resolved_output_dir / f"{split}_errors.jsonl", rows)

    question_bug_rows = question_as_bug(split_errors["test"])
    write_jsonl(resolved_output_dir / "question_as_bug_test.jsonl", question_bug_rows)
    write_json(confusion_summary, resolved_output_dir / "confusion_summary.json")
    write_json(error_summary, resolved_output_dir / "per_class_error_summary.json")
    return {
        "confusion_summary": confusion_summary,
        "output_dir": display_path(resolved_output_dir),
        "per_class_error_summary": error_summary,
        "question_as_bug_test_count": len(question_bug_rows),
        "test_error_count": len(split_errors["test"]),
        "val_error_count": len(split_errors["val"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved DistilBERT classification errors.")
    parser.add_argument("--model-dir", type=Path, default=TrainingConfig().output_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    summary = analyze_transformer_errors(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        split_paths=split_paths,
        max_length=args.max_length,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
