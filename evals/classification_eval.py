from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
LABEL_ORDER: tuple[str, ...] = ("bug", "feature", "docs", "question")
ID_TO_LABEL = {index: label for index, label in enumerate(LABEL_ORDER)}

DEFAULT_GOLDEN_PATH = REPO_ROOT / "data" / "splits" / "golden_eval.jsonl"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "eval_thresholds.yaml"
DEFAULT_REPORT_PATH = (
    REPO_ROOT / "artifacts" / "evals" / "classification_eval_report.json"
)
DEFAULT_CLASSICAL_MODEL_PATH = (
    REPO_ROOT / "artifacts" / "classification" / "classical" / "model.joblib"
)
DEFAULT_DISTILBERT_MANIFEST_PATH = (
    REPO_ROOT / "artifacts" / "classification" / "distilbert_selected_manifest.json"
)
DEFAULT_LLM_CACHE_PATH = (
    REPO_ROOT / "artifacts" / "evals" / "classification_llm_golden_predictions.jsonl"
)
WORKER_ENV = "CLASSIFICATION_EVAL_WORKER"


class Example(Protocol):
    issue_id: str
    label: str
    text: str


class Dataset(Protocol):
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int: ...

    def __iter__(self) -> Any: ...


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    resolved = repo_path(path)
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def ensure_import_paths() -> None:
    for path in (REPO_ROOT, ML_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def reexec_in_ml_environment(argv: Sequence[str]) -> int:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(ML_DIR)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env[WORKER_ENV] = "1"
    command = [
        "uv",
        "run",
        "--project",
        str(ML_DIR),
        "python",
        "-m",
        "evals.classification_eval",
        "--worker",
        *argv,
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


def read_json(path: Path) -> JsonObject:
    resolved = repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        parsed: object = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{display_path(resolved)} must contain a JSON object")
    return cast(JsonObject, parsed)


def write_json(record: JsonObject, path: Path) -> None:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[JsonObject]:
    resolved = repo_path(path)
    if not resolved.exists():
        return []
    rows: list[JsonObject] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(cast(JsonObject, parsed))
    return rows


def append_jsonl(path: Path, row: JsonObject) -> None:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def load_golden_dataset(path: Path) -> Dataset:
    ensure_import_paths()
    from classification.data import load_classification_dataset

    return cast(
        Dataset, load_classification_dataset(repo_path(path), split="golden_eval")
    )


def load_thresholds(path: Path) -> JsonObject:
    thresholds = read_json(path)
    classification = thresholds.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("eval_thresholds.yaml must contain a classification object")
    return thresholds


def empty_confusion_matrix() -> list[list[int]]:
    return [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]


def classification_metrics(
    truth: Sequence[str],
    predictions: Sequence[str],
    *,
    latency_seconds: float | None = None,
) -> JsonObject:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length")

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    matrix = empty_confusion_matrix()
    correct = 0
    for true_label, predicted_label in zip(truth, predictions, strict=True):
        if true_label not in label_to_index:
            raise ValueError(f"unknown true label {true_label!r}")
        if predicted_label not in label_to_index:
            raise ValueError(f"unknown predicted label {predicted_label!r}")
        matrix[label_to_index[true_label]][label_to_index[predicted_label]] += 1
        if true_label == predicted_label:
            correct += 1

    per_class: dict[str, float] = {}
    for label, index in label_to_index.items():
        true_positive = matrix[index][index]
        predicted_total = sum(row[index] for row in matrix)
        actual_total = sum(matrix[index])
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        per_class[label] = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall
            else 0.0
        )

    examples = len(truth)
    metrics: JsonObject = {
        "accuracy": (correct / examples) if examples else 0.0,
        "confusion_matrix": {
            "labels": list(LABEL_ORDER),
            "matrix": matrix,
        },
        "macro_f1": sum(per_class.values()) / len(LABEL_ORDER),
        "per_class_f1": per_class,
    }
    if latency_seconds is not None:
        metrics["latency"] = {
            "examples": examples,
            "ms_per_example": (latency_seconds / examples) * 1000 if examples else None,
            "total_seconds": latency_seconds,
        }
    return metrics


def evaluate_thresholds(
    metrics: JsonObject, thresholds: Mapping[str, object]
) -> JsonObject:
    checks: JsonObject = {}
    passed = True
    for metric_name, threshold in thresholds.items():
        if metric_name == "required" or not isinstance(threshold, int | float):
            continue
        value = metrics.get(metric_name)
        metric_passed = isinstance(value, int | float) and float(value) >= float(
            threshold
        )
        checks[metric_name] = {
            "actual": float(value) if isinstance(value, int | float) else None,
            "passed": metric_passed,
            "threshold": float(threshold),
        }
        passed = passed and metric_passed
    return {
        "checks": checks,
        "passed": passed,
    }


def model_required(thresholds: Mapping[str, object]) -> bool:
    return bool(thresholds.get("required", True))


def model_report(
    *,
    metrics: JsonObject,
    predictions: Sequence[str],
    thresholds: Mapping[str, object],
    status_prefix: str = "evaluated",
    extra: JsonObject | None = None,
) -> JsonObject:
    threshold_result = evaluate_thresholds(metrics, thresholds)
    metrics_passed = bool(threshold_result["passed"])
    required = model_required(thresholds)
    gate_passed = metrics_passed or not required
    threshold_result["metrics_passed"] = metrics_passed
    threshold_result["passed"] = gate_passed
    report: JsonObject = {
        "metrics": metrics,
        "predictions_count": len(predictions),
        "required": required,
        "status": "pass"
        if metrics_passed
        else ("optional_fail" if not required else "fail"),
        "status_detail": status_prefix,
        "thresholds": threshold_result,
    }
    if extra:
        report.update(extra)
    return report


def skipped_report(*, reason: str, thresholds: Mapping[str, object]) -> JsonObject:
    required = model_required(thresholds)
    return {
        "required": required,
        "skip_reason": reason,
        "status": "fail" if required else "skipped",
        "thresholds": {
            "checks": {},
            "passed": not required,
        },
    }


def unavailable_report(
    *,
    reason: str,
    thresholds: Mapping[str, object],
    artifact_path: Path | None = None,
) -> JsonObject:
    report = skipped_report(reason=reason, thresholds=thresholds)
    report["status_detail"] = (
        "required artifact unavailable"
        if report["required"]
        else "optional artifact unavailable"
    )
    if artifact_path is not None:
        report["artifact_path"] = display_path(artifact_path)
    return report


def evaluate_classical(
    dataset: Dataset, model_path: Path, thresholds: Mapping[str, object]
) -> JsonObject:
    import joblib

    resolved_model_path = repo_path(model_path)
    if not resolved_model_path.exists():
        return unavailable_report(
            reason=f"required classical model artifact is missing: {display_path(model_path)}",
            thresholds=thresholds,
            artifact_path=model_path,
        )

    started = time.perf_counter()
    model = joblib.load(resolved_model_path)
    predictions = [str(label) for label in model.predict(dataset.texts)]
    latency = time.perf_counter() - started
    metrics = classification_metrics(
        dataset.labels, predictions, latency_seconds=latency
    )
    return model_report(
        metrics=metrics,
        predictions=predictions,
        thresholds=thresholds,
        extra={"model_path": display_path(model_path)},
    )


def distilbert_model_dir(manifest_path: Path) -> Path:
    manifest = read_json(manifest_path)
    selected = manifest.get("selected_model_dir")
    if not isinstance(selected, str) or not selected:
        raise ValueError("DistilBERT selected manifest is missing selected_model_dir")
    return repo_path(Path(selected))


def distilbert_artifact_unavailable(manifest_path: Path) -> tuple[bool, str, Path]:
    resolved_manifest_path = repo_path(manifest_path)
    if not resolved_manifest_path.exists():
        return (
            True,
            f"DistilBERT selected manifest is missing: {display_path(manifest_path)}",
            manifest_path,
        )

    model_dir = distilbert_model_dir(manifest_path)
    if not model_dir.exists():
        return (
            True,
            f"DistilBERT selected model directory is missing: {display_path(model_dir)}",
            model_dir,
        )

    has_model_weights = (model_dir / "model.safetensors").exists() or (
        model_dir / "pytorch_model.bin"
    ).exists()
    if not has_model_weights:
        return (
            True,
            f"DistilBERT model weights are missing in {display_path(model_dir)}",
            model_dir,
        )

    return False, "", model_dir


def max_length_for_distilbert(model_dir: Path) -> int:
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        return 256
    config = read_json(config_path)
    value = config.get("max_length")
    return int(value) if isinstance(value, int) else 256


def evaluate_distilbert(
    dataset: Dataset,
    manifest_path: Path,
    thresholds: Mapping[str, object],
    *,
    batch_size: int = 8,
) -> JsonObject:
    unavailable, reason, artifact_path = distilbert_artifact_unavailable(manifest_path)
    if unavailable:
        return unavailable_report(
            reason=reason,
            thresholds=thresholds,
            artifact_path=artifact_path,
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = distilbert_model_dir(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = max_length_for_distilbert(model_dir)

    predictions: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(dataset.texts), batch_size):
            texts = dataset.texts[start : start + batch_size]
            encoded = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            predictions.extend(ID_TO_LABEL[int(index)] for index in predicted_ids)
    latency = time.perf_counter() - started
    metrics = classification_metrics(
        dataset.labels, predictions, latency_seconds=latency
    )
    return model_report(
        metrics=metrics,
        predictions=predictions,
        thresholds=thresholds,
        extra={
            "device": str(device),
            "model_dir": display_path(model_dir),
        },
    )


def cached_llm_predictions(
    dataset: Dataset, cache_path: Path
) -> tuple[list[str] | None, str]:
    rows = read_jsonl(cache_path)
    by_issue_id = {
        str(row["issue_id"]): row
        for row in rows
        if isinstance(row.get("issue_id"), str)
        and row.get("predicted_label") in LABEL_ORDER
        and not row.get("error")
    }

    predictions: list[str] = []
    missing: list[str] = []
    for example in dataset:
        row = by_issue_id.get(example.issue_id)
        if row is None:
            missing.append(example.issue_id)
            continue
        predictions.append(str(row["predicted_label"]))

    if missing:
        return (
            None,
            f"cached LLM predictions cover {len(predictions)}/{len(dataset)} golden examples",
        )
    return predictions, "using cached golden LLM predictions"


def fill_llm_cache_with_api(dataset: Dataset, cache_path: Path) -> None:
    ensure_import_paths()
    from classification.evaluate_llm_baseline import (
        AnthropicIssueClassifier,
        append_jsonl as append_llm_jsonl,
        load_prompt,
        prediction_row,
        prompt_sha256,
        render_prompt,
    )
    from classification.llm_secret_loader import load_llm_api_key

    prompt_path = ML_DIR / "classification" / "llm_baseline_prompt.md"
    prompt = load_prompt(prompt_path)
    prompt_hash = prompt_sha256(prompt_path)
    secret = load_llm_api_key()
    classifier = AnthropicIssueClassifier(
        api_key=secret.api_key, model="claude-haiku-4-5"
    )

    existing_rows = read_jsonl(cache_path)
    completed = {
        str(row["issue_id"])
        for row in existing_rows
        if row.get("predicted_label") in LABEL_ORDER and not row.get("error")
    }
    for example in dataset:
        if example.issue_id in completed:
            continue
        started = time.perf_counter()
        try:
            result = classifier.classify(render_prompt(prompt, example))
            row = prediction_row(
                example=example,
                model=classifier.model,
                prompt_hash=prompt_hash,
                result=result,
            )
        except Exception as exc:
            row = prediction_row(
                example=example,
                model=classifier.model,
                prompt_hash=prompt_hash,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        append_llm_jsonl(repo_path(cache_path), row)


def evaluate_llm_cached_or_allowed(
    dataset: Dataset,
    cache_path: Path,
    thresholds: Mapping[str, object],
    *,
    allow_api: bool,
) -> JsonObject:
    if allow_api:
        fill_llm_cache_with_api(dataset, cache_path)

    predictions, reason = cached_llm_predictions(dataset, cache_path)
    if predictions is None:
        if allow_api:
            return skipped_report(
                reason=f"LLM API run did not produce complete golden cache: {reason}",
                thresholds=thresholds,
            )
        return skipped_report(
            reason=f"{reason}; pass --allow-llm-api to spend new API calls",
            thresholds=thresholds,
        )

    metrics = classification_metrics(dataset.labels, predictions)
    return model_report(
        metrics=metrics,
        predictions=predictions,
        thresholds=thresholds,
        status_prefix=reason,
        extra={"cache_path": display_path(cache_path)},
    )


def model_thresholds(
    all_thresholds: JsonObject, model_name: str
) -> Mapping[str, object]:
    classification = all_thresholds["classification"]
    if not isinstance(classification, dict):
        raise ValueError("classification thresholds must be an object")
    value = classification.get(model_name)
    if not isinstance(value, dict):
        raise ValueError(f"thresholds missing classification.{model_name}")
    return cast(Mapping[str, object], value)


def overall_pass(models: Mapping[str, JsonObject]) -> bool:
    return all(
        bool(model.get("thresholds", {}).get("passed")) for model in models.values()
    )


def failed_eval_report(
    *,
    reason: str,
    golden_path: Path,
    thresholds_path: Path,
    thresholds: JsonObject,
    output_path: Path,
) -> JsonObject:
    report: JsonObject = {
        "generated_at": datetime.now(UTC).isoformat(),
        "golden_examples": 0,
        "inputs": {
            "golden_eval": display_path(golden_path),
            "thresholds": display_path(thresholds_path),
        },
        "models": {},
        "overall_pass": False,
        "status": "fail",
        "failure_reason": reason,
        "thresholds": thresholds,
    }
    write_json(report, output_path)
    return report


def run_classification_eval(
    *,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
    output_path: Path = DEFAULT_REPORT_PATH,
    classical_model_path: Path = DEFAULT_CLASSICAL_MODEL_PATH,
    distilbert_manifest_path: Path = DEFAULT_DISTILBERT_MANIFEST_PATH,
    llm_cache_path: Path = DEFAULT_LLM_CACHE_PATH,
    allow_llm_api: bool = False,
) -> JsonObject:
    thresholds = load_thresholds(thresholds_path)
    if not repo_path(golden_path).exists():
        return failed_eval_report(
            reason=f"required golden eval fixture is missing: {display_path(golden_path)}",
            golden_path=golden_path,
            output_path=output_path,
            thresholds=thresholds,
            thresholds_path=thresholds_path,
        )

    dataset = load_golden_dataset(golden_path)
    models = {
        "classical_ml": evaluate_classical(
            dataset,
            classical_model_path,
            model_thresholds(thresholds, "classical_ml"),
        ),
        "selected_distilbert": evaluate_distilbert(
            dataset,
            distilbert_manifest_path,
            model_thresholds(thresholds, "selected_distilbert"),
        ),
        "llm_baseline": evaluate_llm_cached_or_allowed(
            dataset,
            llm_cache_path,
            model_thresholds(thresholds, "llm_baseline"),
            allow_api=allow_llm_api,
        ),
    }
    report: JsonObject = {
        "allow_llm_api": allow_llm_api,
        "generated_at": datetime.now(UTC).isoformat(),
        "golden_examples": len(dataset),
        "inputs": {
            "classical_model": display_path(classical_model_path),
            "distilbert_manifest": display_path(distilbert_manifest_path),
            "golden_eval": display_path(golden_path),
            "llm_cache": display_path(llm_cache_path),
            "thresholds": display_path(thresholds_path),
        },
        "models": models,
        "overall_pass": overall_pass(models),
        "thresholds": thresholds,
    }
    write_json(report, output_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification golden eval.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--classical-model-path", type=Path, default=DEFAULT_CLASSICAL_MODEL_PATH
    )
    parser.add_argument(
        "--distilbert-manifest-path",
        type=Path,
        default=DEFAULT_DISTILBERT_MANIFEST_PATH,
    )
    parser.add_argument("--llm-cache-path", type=Path, default=DEFAULT_LLM_CACHE_PATH)
    parser.add_argument(
        "--allow-llm-api",
        action="store_true",
        help="Spend LLM API calls for missing golden predictions.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--worker" not in raw_args and os.environ.get(WORKER_ENV) != "1":
        return reexec_in_ml_environment(raw_args)

    ensure_import_paths()
    args = parse_args(raw_args)
    report = run_classification_eval(
        allow_llm_api=args.allow_llm_api,
        classical_model_path=args.classical_model_path,
        distilbert_manifest_path=args.distilbert_manifest_path,
        golden_path=args.golden_path,
        llm_cache_path=args.llm_cache_path,
        output_path=args.output_path,
        thresholds_path=args.thresholds_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
