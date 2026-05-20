from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from classification.data import LABEL_ORDER, display_path, resolve_repo_path
from classification.training_config import TrainingConfig

JsonObject = dict[str, Any]
DEFAULT_BASELINE_DIR = TrainingConfig().resolved_output_dir
DEFAULT_OUTPUT_PATH = DEFAULT_BASELINE_DIR / "experiment_comparison.json"
DEFAULT_EXPERIMENT_DIRS: tuple[Path, ...] = (
    TrainingConfig().resolved_output_dir.parent / "distilbert_exp_epochs2",
    TrainingConfig().resolved_output_dir.parent / "distilbert_exp_len384_epochs2",
    TrainingConfig().resolved_output_dir.parent / "distilbert_exp_lr1e5",
)
MACRO_F1_SIMILARITY_DELTA = 0.005
QUESTION_F1_SIMILARITY_DELTA = 0.005


def read_json(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        parsed: object = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{display_path(path)} must contain a JSON object")
    return parsed


def split_summary(metrics: JsonObject) -> JsonObject:
    per_class_f1 = metrics.get("per_class_f1", {})
    if not isinstance(per_class_f1, dict):
        per_class_f1 = {}
    return {
        "accuracy": metrics.get("accuracy"),
        "latency": metrics.get("latency"),
        "macro_f1": metrics.get("macro_f1"),
        "per_class_f1": {label: per_class_f1.get(label) for label in LABEL_ORDER},
        "question_f1": per_class_f1.get("question"),
    }


def config_summary(config: JsonObject) -> JsonObject:
    return {
        "epochs": config.get("epochs"),
        "learning_rate": config.get("learning_rate"),
        "max_length": config.get("max_length"),
        "seed": config.get("seed"),
    }


def candidate_from_training_dir(
    *,
    name: str,
    kind: str,
    path: Path,
) -> JsonObject | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    config_path = path / "training_config.json"
    config = read_json(config_path) if config_path.exists() else {}
    test_metrics_path = path / "test_metrics.json"
    test_metrics = read_json(test_metrics_path) if test_metrics_path.exists() else metrics["test"]
    model = metrics.get("model", {})
    train = metrics.get("train", {})
    if not isinstance(model, dict):
        model = {}
    if not isinstance(train, dict):
        train = {}
    return {
        "artifact_sha256": model.get("artifact_sha256"),
        "config": config_summary(config),
        "kind": kind,
        "name": name,
        "output_dir": display_path(path),
        "test": split_summary(test_metrics),
        "train_runtime_seconds": train.get("seconds"),
        "validation": split_summary(metrics["val"]),
    }


def candidate_from_checkpoint(entry: JsonObject) -> JsonObject:
    return {
        "artifact_sha256": None,
        "config": {},
        "kind": "checkpoint",
        "name": entry["name"],
        "output_dir": entry["checkpoint_path"],
        "test": entry["test"],
        "train_runtime_seconds": None,
        "validation": entry["validation"],
    }


def metric_value(candidate: JsonObject, split: str, metric: str) -> float:
    value = candidate[split][metric]
    if not isinstance(value, int | float):
        return 0.0
    return float(value)


def latency_value(candidate: JsonObject) -> float:
    latency = candidate["test"].get("latency")
    if not isinstance(latency, dict):
        return float("inf")
    value = latency.get("ms_per_example")
    if not isinstance(value, int | float):
        return float("inf")
    return float(value)


def config_value(candidate: JsonObject, field: str) -> float:
    config = candidate.get("config", {})
    if not isinstance(config, dict):
        return float("inf")
    value = config.get(field)
    if not isinstance(value, int | float):
        return float("inf")
    return float(value)


def prefer_candidate(candidate: JsonObject, incumbent: JsonObject) -> bool:
    candidate_macro = metric_value(candidate, "test", "macro_f1")
    incumbent_macro = metric_value(incumbent, "test", "macro_f1")
    if candidate_macro > incumbent_macro + MACRO_F1_SIMILARITY_DELTA:
        return True
    if incumbent_macro > candidate_macro + MACRO_F1_SIMILARITY_DELTA:
        return False

    candidate_question = metric_value(candidate, "test", "question_f1")
    incumbent_question = metric_value(incumbent, "test", "question_f1")
    if candidate_question > incumbent_question + QUESTION_F1_SIMILARITY_DELTA:
        return True
    if incumbent_question > candidate_question + QUESTION_F1_SIMILARITY_DELTA:
        return False

    candidate_cost = (
        latency_value(candidate),
        config_value(candidate, "epochs"),
        config_value(candidate, "max_length"),
    )
    incumbent_cost = (
        latency_value(incumbent),
        config_value(incumbent, "epochs"),
        config_value(incumbent, "max_length"),
    )
    return candidate_cost < incumbent_cost


def select_best_candidate(candidates: list[JsonObject]) -> JsonObject:
    if not candidates:
        raise ValueError("at least one candidate is required")
    best = candidates[0]
    for candidate in candidates[1:]:
        if prefer_candidate(candidate, best):
            best = candidate
    return best


def experiment_candidates(
    *,
    baseline_dir: Path,
    experiment_dirs: list[Path],
    checkpoint_comparison_path: Path,
) -> list[JsonObject]:
    candidates: list[JsonObject] = []
    baseline = candidate_from_training_dir(
        name="current_distilbert",
        kind="baseline",
        path=baseline_dir,
    )
    if baseline is not None:
        candidates.append(baseline)

    if checkpoint_comparison_path.exists():
        checkpoint_comparison = read_json(checkpoint_comparison_path)
        for entry in checkpoint_comparison.get("candidates", []):
            if isinstance(entry, dict) and entry.get("name") != "final":
                candidates.append(candidate_from_checkpoint(entry))

    for path in experiment_dirs:
        experiment = candidate_from_training_dir(name=path.name, kind="experiment", path=path)
        if experiment is not None:
            candidates.append(experiment)
    return candidates


def compare_transformer_experiments(
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    experiment_dirs: list[Path] | None = None,
    checkpoint_comparison_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> JsonObject:
    resolved_baseline_dir = resolve_repo_path(baseline_dir)
    resolved_experiment_dirs = [
        resolve_repo_path(path) for path in (experiment_dirs or list(DEFAULT_EXPERIMENT_DIRS))
    ]
    if checkpoint_comparison_path is None:
        checkpoint_comparison_path = resolved_baseline_dir / "checkpoint_comparison.json"
    resolved_checkpoint_path = resolve_repo_path(checkpoint_comparison_path)
    candidates = experiment_candidates(
        baseline_dir=resolved_baseline_dir,
        experiment_dirs=resolved_experiment_dirs,
        checkpoint_comparison_path=resolved_checkpoint_path,
    )
    best = select_best_candidate(candidates)
    comparison: JsonObject = {
        "best_candidate": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_rule": {
            "macro_f1_similarity_delta": MACRO_F1_SIMILARITY_DELTA,
            "primary": "higher test macro_f1",
            "question_f1_similarity_delta": QUESTION_F1_SIMILARITY_DELTA,
            "secondary": "higher test question_f1 when macro_f1 is similar",
            "tertiary": "lower test latency, epochs, then max_length when performance is similar",
        },
    }
    resolved_output_path = resolve_repo_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DistilBERT controlled experiments.")
    parser.add_argument("--baseline-dir", type=Path, default=TrainingConfig().output_dir)
    parser.add_argument("--checkpoint-comparison-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--experiment-dir", action="append", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_transformer_experiments(
        baseline_dir=args.baseline_dir,
        checkpoint_comparison_path=args.checkpoint_comparison_path,
        experiment_dirs=args.experiment_dir,
        output_path=args.output_path,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
