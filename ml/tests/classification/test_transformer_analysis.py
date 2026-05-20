from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from classification.analyze_transformer_errors import (
    per_class_error_summary,
    prediction_error_rows,
    question_as_bug,
)
from classification.compare_transformer_checkpoints import checkpoint_step, discover_checkpoints
from classification.compare_transformer_experiments import select_best_candidate
from classification.data import REPO_ROOT, ClassificationDataset, ClassificationExample
from classification.evaluate_transformer import max_length_for_model_dir

JsonObject = dict[str, Any]


def _example(
    index: int, *, label: str, label_id: int, text: str | None = None
) -> ClassificationExample:
    return ClassificationExample(
        created_at=datetime(2025, 1, index, tzinfo=UTC),
        html_url=f"https://github.com/apache/airflow/issues/{index}",
        issue_id=str(10_000 + index),
        label=label,
        label_id=label_id,
        number=index,
        raw={},
        split="test",
        text=text or f"{label} text",
    )


def test_error_analysis_rows_include_required_shape() -> None:
    dataset = ClassificationDataset(
        split="test",
        examples=(
            _example(1, label="question", label_id=3, text="How do I configure Airflow?"),
            _example(2, label="bug", label_id=0, text="Scheduler crash"),
        ),
    )
    logits = np.array(
        [
            [4.0, 0.1, 0.1, 1.0],
            [3.0, 0.1, 0.1, 0.1],
        ]
    )

    rows = prediction_error_rows(dataset=dataset, logits=logits, split="test")

    assert len(rows) == 1
    assert set(rows[0]) == {
        "confidence",
        "created_at",
        "html_url",
        "issue_id",
        "number",
        "predicted_label",
        "probabilities",
        "split",
        "text_preview",
        "true_label",
    }
    assert rows[0]["true_label"] == "question"
    assert rows[0]["predicted_label"] == "bug"
    assert set(rows[0]["probabilities"]) == {"bug", "feature", "docs", "question"}
    assert question_as_bug(rows) == rows


def test_per_class_error_summary_counts_prediction_patterns() -> None:
    dataset = ClassificationDataset(
        split="test",
        examples=(
            _example(1, label="question", label_id=3),
            _example(2, label="bug", label_id=0),
        ),
    )
    rows: list[JsonObject] = [
        {
            "predicted_label": "bug",
            "true_label": "question",
        }
    ]

    summary = per_class_error_summary(rows, dataset)

    assert summary["question"]["error_count"] == 1
    assert summary["question"]["total"] == 1
    assert summary["question"]["predicted_as"]["bug"] == 1
    assert summary["bug"]["error_count"] == 0


def test_checkpoint_discovery_sorts_by_numeric_step(tmp_path: Path) -> None:
    for name in ("checkpoint-210", "checkpoint-7", "checkpoint-final", "checkpoint-140"):
        (tmp_path / name).mkdir()

    checkpoints = discover_checkpoints(tmp_path)

    assert [path.name for path in checkpoints] == [
        "checkpoint-7",
        "checkpoint-140",
        "checkpoint-210",
    ]
    assert checkpoint_step(tmp_path / "checkpoint-final") == -1


def test_experiment_selection_prefers_macro_then_question_then_speed() -> None:
    incumbent: JsonObject = {
        "config": {"epochs": 3, "max_length": 256},
        "name": "baseline",
        "test": {
            "latency": {"ms_per_example": 8.0},
            "macro_f1": 0.65,
            "question_f1": 0.32,
        },
    }
    better_question: JsonObject = {
        "config": {"epochs": 2, "max_length": 384},
        "name": "better_question",
        "test": {
            "latency": {"ms_per_example": 9.0},
            "macro_f1": 0.652,
            "question_f1": 0.39,
        },
    }
    better_macro: JsonObject = {
        "config": {"epochs": 2, "max_length": 256},
        "name": "better_macro",
        "test": {
            "latency": {"ms_per_example": 8.5},
            "macro_f1": 0.67,
            "question_f1": 0.31,
        },
    }

    assert select_best_candidate([incumbent, better_question])["name"] == "better_question"
    assert (
        select_best_candidate([incumbent, better_question, better_macro])["name"] == "better_macro"
    )


def test_experiment_selection_uses_latency_when_metrics_are_similar() -> None:
    slower: JsonObject = {
        "config": {"epochs": 3, "max_length": 256},
        "name": "slower",
        "test": {
            "latency": {"ms_per_example": 9.0},
            "macro_f1": 0.65,
            "question_f1": 0.32,
        },
    }
    faster: JsonObject = {
        "config": {"epochs": 2, "max_length": 256},
        "name": "faster",
        "test": {
            "latency": {"ms_per_example": 7.0},
            "macro_f1": 0.651,
            "question_f1": 0.321,
        },
    }

    assert select_best_candidate([slower, faster])["name"] == "faster"


def test_max_length_for_model_dir_reads_saved_training_config(tmp_path: Path) -> None:
    (tmp_path / "training_config.json").write_text(
        json.dumps({"max_length": 384}),
        encoding="utf-8",
    )

    assert max_length_for_model_dir(tmp_path) == 384
    assert max_length_for_model_dir(tmp_path, explicit_max_length=128) == 128


def test_no_phase_7_or_endpoint_code_added() -> None:
    forbidden_paths = [
        REPO_ROOT / "model-server" / "app" / "classification.py",
        REPO_ROOT / "model-server" / "app" / "ner.py",
        REPO_ROOT / "model-server" / "app" / "summarization.py",
    ]

    assert all(not path.exists() for path in forbidden_paths)
