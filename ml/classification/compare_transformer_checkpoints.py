from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classification.data import (
    ClassificationSplitPaths,
    display_path,
    load_classification_dataset,
    resolve_repo_path,
)
from classification.evaluate_transformer import (
    dataset_path_for_split,
    evaluate_transformer_dataset,
    max_length_for_model_dir,
)
from classification.training_config import TrainingConfig

JsonObject = dict[str, Any]
DEFAULT_MODEL_DIR = TrainingConfig().resolved_output_dir
DEFAULT_OUTPUT_PATH = DEFAULT_MODEL_DIR / "checkpoint_comparison.json"


def checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix):
        return -1
    try:
        return int(path.name.removeprefix(prefix))
    except ValueError:
        return -1


def discover_checkpoints(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    checkpoints = [
        path
        for path in model_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-") and checkpoint_step(path) >= 0
    ]
    return sorted(checkpoints, key=checkpoint_step)


def compact_metrics(metrics: JsonObject) -> JsonObject:
    return {
        "accuracy": metrics["accuracy"],
        "latency": metrics.get("latency"),
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "question_f1": metrics["per_class_f1"]["question"],
    }


def evaluate_candidate(
    *,
    candidate_dir: Path,
    name: str,
    split_paths: ClassificationSplitPaths,
    output_dir: Path,
    max_length: int,
) -> JsonObject:
    tokenizer = AutoTokenizer.from_pretrained(candidate_dir)
    model = AutoModelForSequenceClassification.from_pretrained(candidate_dir)
    result: JsonObject = {
        "checkpoint_path": display_path(candidate_dir),
        "name": name,
    }
    for split in ("val", "test"):
        dataset = load_classification_dataset(
            dataset_path_for_split(split, split_paths), split=split
        )
        tokenized = tokenize_dataset_for_candidate(
            tokenizer=tokenizer,
            dataset=dataset,
            max_length=max_length,
        )
        metrics = evaluate_transformer_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=tokenized,
            split=split,
            output_dir=output_dir / name / split,
            model_dir=candidate_dir,
        )
        result["validation" if split == "val" else "test"] = compact_metrics(metrics)
    return result


def tokenize_dataset_for_candidate(
    *,
    tokenizer: Any,
    dataset: Any,
    max_length: int,
) -> Any:
    from classification.train_transformer import tokenize_dataset

    return tokenize_dataset(tokenizer, dataset, max_length=max_length)


def compare_transformer_checkpoints(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    split_paths: ClassificationSplitPaths | None = None,
    max_length: int | None = None,
) -> JsonObject:
    if split_paths is None:
        split_paths = ClassificationSplitPaths()
    resolved_model_dir = resolve_repo_path(model_dir)
    resolved_output_path = resolve_repo_path(output_path)
    resolved_max_length = max_length_for_model_dir(resolved_model_dir, max_length)
    candidates = [("final", resolved_model_dir)] + [
        (path.name, path) for path in discover_checkpoints(resolved_model_dir)
    ]
    eval_dir = resolved_model_dir / "checkpoint_comparison_eval"
    results = [
        evaluate_candidate(
            candidate_dir=path,
            name=name,
            split_paths=split_paths,
            output_dir=eval_dir,
            max_length=resolved_max_length,
        )
        for name, path in candidates
    ]
    comparison: JsonObject = {
        "candidate_count": len(results),
        "checkpoint_count": len(candidates) - 1,
        "candidates": results,
        "max_length": resolved_max_length,
        "model_dir": display_path(resolved_model_dir),
    }
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved DistilBERT checkpoints.")
    parser.add_argument("--model-dir", type=Path, default=TrainingConfig().output_dir)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
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
    comparison = compare_transformer_checkpoints(
        model_dir=args.model_dir,
        output_path=args.output_path,
        split_paths=split_paths,
        max_length=args.max_length,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
