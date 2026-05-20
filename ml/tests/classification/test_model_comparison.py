from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classification.compare_models import (
    CLASSICAL_MODEL_NAME,
    ComparisonPaths,
    compare_models,
    comparison_payload,
)

JsonObject = dict[str, Any]


def _write_json(path: Path, record: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _classical_metrics() -> JsonObject:
    return {
        "accuracy": 0.72,
        "latency": {"examples": 100, "ms_per_example": 0.4},
        "macro_f1": 0.71,
        "model": {"size_bytes": 1000},
        "per_class_f1": {
            "bug": 0.7,
            "docs": 0.8,
            "feature": 0.85,
            "question": 0.5,
        },
    }


def _distilbert_manifest(tmp_path: Path) -> JsonObject:
    model_dir = tmp_path / "distilbert"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"model")
    _write_json(model_dir / "test_metrics.json", {"latency": {"examples": 100}})
    return {
        "latency_ms_per_example": 8.0,
        "per_class_f1": {
            "bug": 0.56,
            "docs": 0.85,
            "feature": 0.86,
            "question": 0.39,
        },
        "selected_model_dir": str(model_dir),
        "test_accuracy": 0.68,
        "test_macro_f1": 0.66,
    }


def _llm_metrics() -> JsonObject:
    return {
        "accuracy": 0.65,
        "average_latency_ms": 1400.0,
        "estimated_cost_usd": None,
        "macro_f1": 0.60,
        "model": "claude-haiku-4-5",
        "per_class_f1": {
            "bug": 0.58,
            "docs": 0.82,
            "feature": 1.0,
            "question": 0.0,
        },
        "sampling_mode": "limit_per_class",
        "selected_per_class": {
            "bug": 10,
            "docs": 10,
            "feature": 10,
            "question": 10,
        },
        "token_usage": {"total_tokens": 100},
        "total_examples_attempted": 40,
    }


def test_comparison_recommends_classical_baseline(tmp_path: Path) -> None:
    payload = comparison_payload(
        classical_metrics=_classical_metrics(),
        distilbert_manifest=_distilbert_manifest(tmp_path),
        llm_metrics=_llm_metrics(),
        paths=ComparisonPaths(),
    )

    assert payload["recommendation"]["deployment_choice"] == CLASSICAL_MODEL_NAME
    assert [row["name"] for row in payload["models"]] == [
        "classical_ml",
        "selected_distilbert",
        "llm_baseline",
    ]
    assert payload["models"][2]["evaluation_scope"] == "balanced sample (40 examples)"


def test_compare_models_writes_json_markdown_and_decision(tmp_path: Path) -> None:
    classical_path = tmp_path / "classical.json"
    distilbert_path = tmp_path / "distilbert_manifest.json"
    llm_path = tmp_path / "llm.json"
    output_path = tmp_path / "model_comparison.json"
    report_path = tmp_path / "model_comparison.md"
    decisions_path = tmp_path / "DECISIONS.md"
    _write_json(classical_path, _classical_metrics())
    _write_json(distilbert_path, _distilbert_manifest(tmp_path))
    _write_json(llm_path, _llm_metrics())
    decisions_path.write_text("# DECISIONS.md\n", encoding="utf-8")

    compare_models(
        ComparisonPaths(
            classical_metrics_path=classical_path,
            distilbert_manifest_path=distilbert_path,
            llm_metrics_path=llm_path,
            output_path=output_path,
            report_path=report_path,
            decisions_path=decisions_path,
        )
    )

    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["recommendation"]["deployment_choice"]
        == "classical_ml"
    )
    assert "Classification Model Comparison" in report_path.read_text(encoding="utf-8")
    assert "D2.2 Classification deployment candidate" in decisions_path.read_text(encoding="utf-8")
