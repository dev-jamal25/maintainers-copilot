from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import summarization


class FakeTensor:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> FakeTensor:
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, FakeTensor]:
        self.calls.append(
            {
                "max_length": max_length,
                "return_tensors": return_tensors,
                "text": text,
                "truncation": truncation,
            }
        )
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    def decode(self, output_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert output_ids == [1, 2, 3]
        assert skip_special_tokens is True
        return "Airflow issue summary."


class FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.eval_called = False
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        return [[1, 2, 3]]


@pytest.fixture(autouse=True)
def clear_summarization_component_cache() -> None:
    summarization.get_summarization_components.cache_clear()


def patch_components(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeTokenizer, FakeModel]:
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()
    monkeypatch.setattr(summarization, "load_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr(summarization, "load_model", lambda: fake_model)
    monkeypatch.setattr(summarization, "inference_device", lambda: "cpu")
    return fake_tokenizer, fake_model


def test_summarize_returns_summary_from_mocked_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tokenizer, fake_model = patch_components(monkeypatch)

    response = summarization.summarize_text(
        "Long Apache Airflow issue thread.",
        max_length=80,
        min_length=20,
    )

    assert response.model == summarization.SUMMARIZATION_MODEL_NAME
    assert response.summary == "Airflow issue summary."
    assert fake_model.device == "cpu"
    assert fake_model.eval_called is True
    assert fake_tokenizer.calls == [
        {
            "max_length": summarization.MAX_MODEL_INPUT_TOKENS,
            "return_tensors": "pt",
            "text": "Long Apache Airflow issue thread.",
            "truncation": True,
        }
    ]
    assert fake_model.generate_calls[0]["max_length"] == 80
    assert fake_model.generate_calls[0]["min_length"] == 20


def test_summarization_components_are_cached_between_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models: list[FakeModel] = []

    fake_tokenizer = FakeTokenizer()

    def fake_model_factory() -> FakeModel:
        model = FakeModel()
        created_models.append(model)
        return model

    monkeypatch.setattr(summarization, "load_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr(summarization, "load_model", fake_model_factory)
    monkeypatch.setattr(summarization, "inference_device", lambda: "cpu")

    summarization.summarize_text("Long Apache Airflow issue thread.")
    summarization.summarize_text("Long Apache Airflow issue thread.")

    assert len(created_models) == 1
    assert len(created_models[0].generate_calls) == 2


def test_summarization_endpoint_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_components(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/summarize",
        json={
            "text": "Long Apache Airflow issue thread.",
            "max_length": 80,
            "min_length": 20,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Airflow issue summary.",
        "model": summarization.SUMMARIZATION_MODEL_NAME,
    }


def test_summarization_rejects_empty_text() -> None:
    client = TestClient(app)

    response = client.post(
        "/summarize",
        json={"text": "   ", "max_length": 80, "min_length": 20},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("max_length", "min_length"),
    [
        (20, 20),
        (20, 30),
        (9, 5),
        (513, 20),
        (80, 4),
    ],
)
def test_summarization_rejects_invalid_min_max_lengths(
    max_length: int,
    min_length: int,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/summarize",
        json={
            "text": "Long Apache Airflow issue thread.",
            "max_length": max_length,
            "min_length": min_length,
        },
    )

    assert response.status_code == 422
