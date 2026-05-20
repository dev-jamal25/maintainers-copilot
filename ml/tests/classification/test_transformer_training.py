from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from torch import nn

import classification.train_transformer as train_transformer
from classification.data import ClassificationDataset, ClassificationExample
from classification.train_transformer import (
    apply_freeze_policy,
    build_data_collator,
    label_map,
    load_sequence_classification_model,
    pin_all_seeds,
    prediction_metrics,
    selected_device_info,
    sha256_file,
    tokenize_dataset,
    trainer_compute_metrics,
    write_model_card,
)
from classification.training_config import FreezePolicy, TrainingConfig


class DummyTokenizer:
    def __call__(
        self, texts: list[str], *, max_length: int, truncation: bool
    ) -> dict[str, list[list[int]]]:
        assert truncation is True
        return {
            "attention_mask": [[1, 1] for _ in texts],
            "input_ids": [[min(len(text), max_length), 1] for text in texts],
        }


class DummyDistilBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(2, 2)
        self.transformer = SimpleNamespace(layer=nn.ModuleList([nn.Linear(2, 2) for _ in range(6)]))


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distilbert = DummyDistilBert()


def _example(index: int, *, label: str, label_id: int) -> ClassificationExample:
    return ClassificationExample(
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        html_url=f"https://github.com/apache/airflow/issues/{index}",
        issue_id=str(index),
        label=label,
        label_id=label_id,
        number=index,
        raw={},
        split="train",
        text=f"{label} text",
    )


def _dataset() -> ClassificationDataset:
    return ClassificationDataset(
        split="train",
        examples=(
            _example(1, label="bug", label_id=0),
            _example(2, label="feature", label_id=1),
        ),
    )


def test_tokenize_dataset_builds_torch_dataset() -> None:
    dataset = tokenize_dataset(DummyTokenizer(), _dataset(), max_length=8)

    first = dataset[0]

    assert len(dataset) == 2
    assert first["labels"] == 0
    assert first["input_ids"] == [8, 1]
    assert first["attention_mask"] == [1, 1]


def test_tokenization_defers_padding_to_data_collator() -> None:
    tokenizer = DummyTokenizer()

    dataset = tokenize_dataset(tokenizer, _dataset(), max_length=8)
    collator = build_data_collator(tokenizer)

    assert dataset[0]["input_ids"] == [8, 1]
    assert collator.__class__.__name__ == "DataCollatorWithPadding"


def test_prediction_metrics_include_required_fields() -> None:
    label_ids = np.array([0, 1, 2, 3])
    logits = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.5],
        ]
    )

    metrics = prediction_metrics(label_ids, logits, split="test")

    assert metrics["accuracy"] == 0.75
    assert metrics["split"] == "test"
    assert metrics["confusion_matrix"]["labels"] == ["bug", "feature", "docs", "question"]
    assert set(metrics["per_class_f1"]) == {"bug", "feature", "docs", "question"}


def test_trainer_compute_metrics_uses_evaluate_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    class FakeMetric:
        def __init__(self, name: str) -> None:
            self.name = name

        def compute(
            self,
            *,
            predictions: np.ndarray,
            references: np.ndarray,
            **kwargs: object,
        ) -> dict[str, float]:
            if self.name == "accuracy":
                return {"accuracy": float(np.mean(predictions == references))}
            assert kwargs == {"average": "macro"}
            return {"f1": 0.75}

    def fake_load(name: str) -> FakeMetric:
        loaded.append(name)
        return FakeMetric(name)

    monkeypatch.setattr("classification.train_transformer.evaluate.load", fake_load)
    monkeypatch.setattr("classification.train_transformer._ACCURACY_METRIC", None)
    monkeypatch.setattr("classification.train_transformer._F1_METRIC", None)
    eval_prediction = SimpleNamespace(
        label_ids=np.array([0, 1, 1]),
        predictions=np.array([[2.0, 0.0], [0.0, 3.0], [4.0, 0.0]]),
    )

    metrics = trainer_compute_metrics(eval_prediction)

    assert loaded == ["accuracy", "f1"]
    assert metrics == {"accuracy": 2 / 3, "macro_f1": 0.75}


def test_pin_all_seeds_calls_four_rngs_and_transformers_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def record_random_seed(seed: int) -> None:
        calls.append(("random", seed))

    def record_numpy_seed(seed: int) -> None:
        calls.append(("numpy", seed))

    def record_torch_seed(seed: int) -> None:
        calls.append(("torch", seed))

    def record_torch_cuda_seed(seed: int) -> None:
        calls.append(("torch_cuda", seed))

    def record_transformers_seed(seed: int) -> None:
        calls.append(("transformers", seed))

    monkeypatch.setattr("classification.train_transformer.random.seed", record_random_seed)
    monkeypatch.setattr("classification.train_transformer.np.random.seed", record_numpy_seed)
    monkeypatch.setattr("classification.train_transformer.torch.manual_seed", record_torch_seed)
    monkeypatch.setattr(
        "classification.train_transformer.torch.cuda.manual_seed_all",
        record_torch_cuda_seed,
    )
    monkeypatch.setattr("classification.train_transformer.set_seed", record_transformers_seed)

    pin_all_seeds(123)

    assert calls == [
        ("random", 123),
        ("numpy", 123),
        ("torch", 123),
        ("torch_cuda", 123),
        ("transformers", 123),
    ]


def test_model_initialization_uses_sequence_classification_head_and_label_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> object:
            captured["model_name"] = model_name
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(train_transformer, "AutoModelForSequenceClassification", FakeAutoModel)

    load_sequence_classification_model(TrainingConfig())

    assert captured["model_name"] == "distilbert-base-uncased"
    assert captured["num_labels"] == 4
    assert captured["id2label"] == {0: "bug", 1: "feature", 2: "docs", 3: "question"}
    assert captured["label2id"] == {"bug": 0, "feature": 1, "docs": 2, "question": 3}


def test_training_path_does_not_manually_move_model_to_cuda() -> None:
    source = inspect.getsource(train_transformer.train_transformer)

    assert ".to(" not in source


def test_apply_freeze_policy_freezes_embeddings_and_lower_layers() -> None:
    model = DummyModel()

    apply_freeze_policy(model, FreezePolicy.embeddings_and_lower_layers(2))

    embeddings_frozen = [
        not parameter.requires_grad for parameter in model.distilbert.embeddings.parameters()
    ]
    assert all(embeddings_frozen)
    assert all(
        not parameter.requires_grad
        for parameter in model.distilbert.transformer.layer[0].parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.distilbert.transformer.layer[1].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.distilbert.transformer.layer[2].parameters()
    )


def test_label_map_matches_shared_label_order() -> None:
    assert label_map() == {
        "id_to_label": {"0": "bug", "1": "feature", "2": "docs", "3": "question"},
        "label_to_id": {"bug": 0, "docs": 2, "feature": 1, "question": 3},
    }


def test_write_model_card_and_hash_helpers(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"tiny model")
    model_hash = sha256_file(artifact)
    card_path = tmp_path / "model_card.md"
    metrics: dict[str, Any] = {
        "device": {"device_name": "Unit Test Device"},
        "test": {"macro_f1": 0.5},
        "val": {"macro_f1": 0.6},
    }

    write_model_card(
        config=TrainingConfig(output_dir=tmp_path),
        metrics=metrics,
        model_hash=model_hash,
        path=card_path,
    )

    content = card_path.read_text(encoding="utf-8")
    assert "distilbert-base-uncased" in content
    assert "Unit Test Device" in content
    assert "0.6" in content
    assert model_hash in content


def test_selected_device_info_has_expected_shape() -> None:
    device = selected_device_info()

    assert set(device) == {"cuda_available", "device", "device_name", "torch_cuda_version"}
    assert device["device"] in {"cpu", "cuda"}
