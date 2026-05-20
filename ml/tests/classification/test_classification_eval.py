from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.classification_eval import (  # noqa: E402
    cached_llm_predictions,
    classification_metrics,
    evaluate_classical,
    evaluate_distilbert,
    evaluate_llm_cached_or_allowed,
    evaluate_thresholds,
    load_thresholds,
    model_report,
    overall_pass,
    run_classification_eval,
    skipped_report,
)

JsonObject = dict[str, Any]


class Example:
    def __init__(self, issue_id: str, label: str, text: str = "text") -> None:
        self.issue_id = issue_id
        self.label = label
        self.text = text


class Dataset:
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> object:
        return iter(self.examples)

    @property
    def labels(self) -> list[str]:
        return [example.label for example in self.examples]

    @property
    def texts(self) -> list[str]:
        return [example.text for example in self.examples]


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_classification_metrics_compute_expected_values() -> None:
    metrics = classification_metrics(
        ["bug", "bug", "docs", "question"],
        ["bug", "feature", "docs", "bug"],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.375
    assert metrics["per_class_f1"]["bug"] == 0.5
    assert metrics["per_class_f1"]["docs"] == 1.0
    assert metrics["per_class_f1"]["feature"] == 0.0
    assert metrics["per_class_f1"]["question"] == 0.0
    assert metrics["confusion_matrix"]["matrix"] == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]


def test_thresholds_pass_and_fail_per_metric() -> None:
    result = evaluate_thresholds(
        {"accuracy": 0.6, "macro_f1": 0.4},
        {"accuracy": 0.5, "macro_f1": 0.45, "required": True},
    )

    assert result["passed"] is False
    assert result["checks"]["accuracy"]["passed"] is True
    assert result["checks"]["macro_f1"]["passed"] is False


def test_cached_llm_predictions_require_full_golden_coverage(tmp_path: Path) -> None:
    dataset = Dataset(
        [
            Example("1", "bug"),
            Example("2", "docs"),
        ]
    )
    cache_path = tmp_path / "llm.jsonl"
    _write_jsonl(
        cache_path,
        [
            {"issue_id": "1", "predicted_label": "bug"},
        ],
    )

    predictions, reason = cached_llm_predictions(dataset, cache_path)

    assert predictions is None
    assert "1/2" in reason


def test_cached_llm_predictions_return_ordered_predictions(tmp_path: Path) -> None:
    dataset = Dataset(
        [
            Example("1", "bug"),
            Example("2", "docs"),
        ]
    )
    cache_path = tmp_path / "llm.jsonl"
    _write_jsonl(
        cache_path,
        [
            {"issue_id": "2", "predicted_label": "question"},
            {"issue_id": "1", "predicted_label": "bug"},
        ],
    )

    predictions, reason = cached_llm_predictions(dataset, cache_path)

    assert predictions == ["bug", "question"]
    assert "cached" in reason


def test_skipped_optional_model_passes_gate() -> None:
    report = skipped_report(reason="not cached", thresholds={"required": False, "macro_f1": 0.2})

    assert report["status"] == "skipped"
    assert report["thresholds"]["passed"] is True


def test_threshold_file_loads_classification_block(tmp_path: Path) -> None:
    path = tmp_path / "eval_thresholds.yaml"
    path.write_text(
        json.dumps({"classification": {"classical_ml": {"accuracy": 0.2}}}),
        encoding="utf-8",
    )

    thresholds = load_thresholds(path)

    assert thresholds["classification"]["classical_ml"]["accuracy"] == 0.2


def test_missing_golden_file_fails_clearly_and_writes_report(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "eval_thresholds.yaml"
    output_path = tmp_path / "classification_eval_report.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "classification": {
                    "classical_ml": {"accuracy": 0.2, "required": True},
                    "selected_distilbert": {"accuracy": 0.2, "required": False},
                    "llm_baseline": {"accuracy": 0.2, "required": False},
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_classification_eval(
        golden_path=tmp_path / "missing_golden.jsonl",
        thresholds_path=thresholds_path,
        output_path=output_path,
    )

    written_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["status"] == "fail"
    assert "required golden eval fixture is missing" in report["failure_reason"]
    assert written_report["failure_reason"] == report["failure_reason"]


def test_missing_required_classical_artifact_fails_clearly(tmp_path: Path) -> None:
    dataset = Dataset([Example("1", "bug")])

    report = evaluate_classical(
        dataset,
        tmp_path / "missing_model.joblib",
        {"required": True, "accuracy": 0.5},
    )

    assert report["status"] == "fail"
    assert report["required"] is True
    assert report["thresholds"]["passed"] is False
    assert "classical model artifact is missing" in report["skip_reason"]


def test_missing_optional_distilbert_artifact_is_skipped_and_reported(tmp_path: Path) -> None:
    dataset = Dataset([Example("1", "bug")])

    report = evaluate_distilbert(
        dataset,
        tmp_path / "missing_manifest.json",
        {"required": False, "accuracy": 0.5},
    )

    assert report["status"] == "skipped"
    assert report["required"] is False
    assert report["thresholds"]["passed"] is True
    assert "DistilBERT selected manifest is missing" in report["skip_reason"]


def test_missing_optional_llm_predictions_are_skipped_and_reported(tmp_path: Path) -> None:
    dataset = Dataset([Example("1", "bug")])

    report = evaluate_llm_cached_or_allowed(
        dataset,
        tmp_path / "missing_llm_cache.jsonl",
        {"required": False, "accuracy": 0.2},
        allow_api=False,
    )

    assert report["status"] == "skipped"
    assert report["required"] is False
    assert report["thresholds"]["passed"] is True
    assert "cached LLM predictions cover 0/1" in report["skip_reason"]


def test_overall_pass_respects_required_models_only() -> None:
    required_report = model_report(
        metrics={"accuracy": 0.9, "macro_f1": 0.9},
        predictions=["bug"],
        thresholds={"required": True, "accuracy": 0.8},
    )
    optional_report = model_report(
        metrics={"accuracy": 0.0, "macro_f1": 0.0},
        predictions=["bug"],
        thresholds={"required": False, "accuracy": 0.8},
    )

    assert optional_report["status"] == "optional_fail"
    assert optional_report["thresholds"]["metrics_passed"] is False
    assert optional_report["thresholds"]["passed"] is True
    assert overall_pass({"required": required_report, "optional": optional_report}) is True
