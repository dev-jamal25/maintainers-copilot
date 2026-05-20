from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from classification.data import LABEL_ORDER, ClassificationDataset, ClassificationExample
from classification.evaluate_llm_baseline import (
    DEFAULT_PROMPT_PATH,
    LLMCallResult,
    ParsedLLMResponse,
    balanced_selected_examples,
    completed_prediction_issue_ids,
    compute_metrics,
    parse_args,
    parse_llm_json_response,
    run_evaluation,
    selected_per_class_counts,
)
from classification.llm_secret_loader import LLMSecretError, SecretLookupResult, load_llm_api_key

JsonObject = dict[str, Any]


class FakeVaultV2:
    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def read_secret_version(self, *, mount_point: str, path: str) -> JsonObject:
        assert mount_point == "secret"
        assert path == "maintainers-copilot/anthropic"
        return {"data": {"data": self._data}}


class FakeVaultClient:
    def __init__(self, data: dict[str, str]) -> None:
        self.sys = self
        self.secrets = self
        self.kv = self
        self.v2 = FakeVaultV2(data)

    def read_seal_status(self) -> JsonObject:
        return {"initialized": True, "sealed": False}

    def is_authenticated(self) -> bool:
        return True


class FailingVaultClient:
    def __init__(self) -> None:
        self.sys = self

    def read_seal_status(self) -> JsonObject:
        raise OSError("unreachable")


class FakeClassifier:
    model = "fake-claude"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, prompt: str) -> LLMCallResult:
        assert "Issue text:" in prompt
        self.calls += 1
        return LLMCallResult(
            parsed=ParsedLLMResponse(label="bug", confidence=0.8, reason="mentions a failure"),
            latency_ms=12.5,
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _example(index: int, *, label: str = "bug") -> ClassificationExample:
    return ClassificationExample(
        created_at=datetime(2025, 1, index, tzinfo=UTC),
        html_url=f"https://github.com/apache/airflow/issues/{index}",
        issue_id=str(1000 + index),
        label=label,
        label_id=LABEL_ORDER.index(label),
        number=index,
        raw={},
        split="test",
        text=f"{label} issue text {index}",
    )


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _dataset_for_labels(labels: list[str]) -> ClassificationDataset:
    return ClassificationDataset(
        split="test",
        examples=tuple(_example(index + 1, label=label) for index, label in enumerate(labels)),
    )


def test_vault_first_secret_loading_with_mocked_vault_success() -> None:
    env = {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "dev-only-root-token"}

    def factory(**_: object) -> FakeVaultClient:
        return FakeVaultClient({"api_key": "sk-ant-api03-real-enough-for-test"})

    result = load_llm_api_key(env=env, client_factory=factory)

    assert result.source == "vault"
    assert result.source_detail == "secret/maintainers-copilot/anthropic:api_key"
    assert result.api_key == "sk-ant-api03-real-enough-for-test"


def test_env_fallback_when_vault_fails() -> None:
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-api03-env-test",
        "VAULT_ADDR": "http://localhost:8200",
        "VAULT_TOKEN": "dev-only-root-token",
    }

    def factory(**_: object) -> FailingVaultClient:
        return FailingVaultClient()

    result = load_llm_api_key(env=env, client_factory=factory)

    assert result.source == "env"
    assert result.source_detail == "ANTHROPIC_API_KEY"
    assert result.api_key == "sk-ant-api03-env-test"


def test_secret_loading_fails_clearly_when_vault_and_env_are_missing() -> None:
    with pytest.raises(LLMSecretError) as exc_info:
        load_llm_api_key(env={}, client_factory=lambda **_: FakeVaultClient({}))

    message = str(exc_info.value)
    assert "No usable Anthropic API key found" in message
    assert "secret/maintainers-copilot/anthropic" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "sk-ant" not in message


def test_response_json_parsing_and_label_validation() -> None:
    parsed = parse_llm_json_response(
        'Here is the JSON: {"label": "docs", "confidence": 0.75, "reason": "docs wording"}'
    )

    assert parsed.label == "docs"
    assert parsed.confidence == 0.75
    assert parsed.reason == "docs wording"


def test_invalid_llm_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="response label"):
        parse_llm_json_response(
            '{"label": "support", "confidence": 0.8, "reason": "not in taxonomy"}'
        )


def test_metric_computation_from_mocked_predictions() -> None:
    rows: list[JsonObject] = [
        {
            "latency_ms": 10,
            "predicted_label": "bug",
            "true_label": "bug",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
        {
            "latency_ms": 20,
            "predicted_label": "bug",
            "true_label": "question",
            "usage": {"input_tokens": 15, "output_tokens": 3},
        },
        {
            "error": "timeout",
            "latency_ms": 30,
            "predicted_label": None,
            "true_label": "docs",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    ]

    source = "vault"
    metrics = compute_metrics(
        rows,
        model="fake-claude",
        secret_source=source,
        input_price_usd_per_million=1.0,
        output_price_usd_per_million=5.0,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["successful_predictions"] == 2
    assert metrics["failed_predictions"] == 1
    assert metrics["token_usage"]["total_tokens"] == 30
    assert metrics["estimated_cost_usd"] == pytest.approx(0.00005)
    assert metrics["confusion_matrix"]["labels"] == list(LABEL_ORDER)


def test_balanced_sampling_selects_n_per_class_when_enough_examples_exist() -> None:
    dataset = _dataset_for_labels(
        [
            "bug",
            "bug",
            "bug",
            "feature",
            "feature",
            "feature",
            "docs",
            "docs",
            "docs",
            "question",
            "question",
            "question",
        ]
    )

    selected = balanced_selected_examples(dataset, limit_per_class=2)

    assert selected_per_class_counts(selected) == {
        "bug": 2,
        "feature": 2,
        "docs": 2,
        "question": 2,
    }
    assert [example.label for example in selected] == [
        "bug",
        "bug",
        "feature",
        "feature",
        "docs",
        "docs",
        "question",
        "question",
    ]


def test_balanced_sampling_handles_classes_with_fewer_than_n_examples() -> None:
    dataset = _dataset_for_labels(["bug", "bug", "feature", "docs", "question"])

    selected = balanced_selected_examples(dataset, limit_per_class=3)

    assert selected_per_class_counts(selected) == {
        "bug": 2,
        "feature": 1,
        "docs": 1,
        "question": 1,
    }


def test_limit_per_class_conflicts_with_limit_and_all() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--limit-per-class", "2", "--limit", "5"])

    with pytest.raises(SystemExit):
        parse_args(["--limit-per-class", "2", "--all"])


def test_resume_behavior_skips_existing_completed_predictions(tmp_path: Path) -> None:
    output_dir = tmp_path / "llm"
    predictions_path = output_dir / "predictions.jsonl"
    first = _example(1)
    second = _example(2, label="question")
    _write_jsonl(
        predictions_path,
        [
            {
                "issue_id": first.issue_id,
                "predicted_label": "bug",
                "true_label": first.label,
            }
        ],
    )
    classifier = FakeClassifier()

    result = run_evaluation(
        classifier=classifier,
        dataset=ClassificationDataset(split="test", examples=(first, second)),
        limit=None,
        output_dir=output_dir,
        prompt_hash="abc123",
        prompt_template="Classify into bug feature docs question. Return JSON only.",
        resume=True,
        secret=SecretLookupResult(
            api_key="sk-ant-api03-test",
            source="vault",
            source_detail="secret/maintainers-copilot/anthropic:api_key",
        ),
    )

    assert completed_prediction_issue_ids(predictions_path) == {first.issue_id, second.issue_id}
    assert classifier.calls == 1
    assert result["run_summary"]["resume_skipped_completed"] == 1
    assert result["metrics"]["total_examples_attempted"] == 2


def test_run_summary_records_limit_per_class_sampling(tmp_path: Path) -> None:
    dataset = _dataset_for_labels(["bug", "feature", "docs", "question"])

    result = run_evaluation(
        classifier=FakeClassifier(),
        dataset=dataset,
        limit=None,
        limit_per_class=1,
        output_dir=tmp_path / "llm",
        prompt_hash="abc123",
        prompt_template="Classify into bug feature docs question. Return JSON only.",
        resume=False,
        secret=SecretLookupResult(
            api_key="sk-ant-api03-test",
            source="vault",
            source_detail="secret/maintainers-copilot/anthropic:api_key",
        ),
    )

    assert result["run_summary"]["sampling_mode"] == "limit_per_class"
    assert result["run_summary"]["limit_per_class"] == 1
    assert result["run_summary"]["selected_per_class"] == {
        "bug": 1,
        "feature": 1,
        "docs": 1,
        "question": 1,
    }
    assert result["metrics"]["sampling_mode"] == "limit_per_class"
    assert result["metrics"]["limit_per_class"] == 1
    assert result["metrics"]["selected_per_class"] == {
        "bug": 1,
        "feature": 1,
        "docs": 1,
        "question": 1,
    }


def test_prompt_file_exists_and_includes_all_labels() -> None:
    prompt = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")

    assert "JSON only" in prompt
    for label in LABEL_ORDER:
        assert label in prompt
