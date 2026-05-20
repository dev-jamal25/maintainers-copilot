from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import ner


class FakeNERPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> list[Mapping[str, Any]]:
        self.calls += 1
        return [
            {
                "entity_group": "ORG",
                "score": 0.99,
                "word": "Apache Airflow",
                "start": text.index("Apache Airflow"),
                "end": text.index("Apache Airflow") + len("Apache Airflow"),
            },
            {
                "entity_group": "MISC",
                "score": 0.88,
                "word": "Python",
                "start": text.index("Python"),
                "end": text.index("Python") + len("Python"),
            },
            {
                "entity_group": "ORG",
                "score": 0.77,
                "word": "Apache Airflow",
                "start": text.index("Apache Airflow"),
                "end": text.index("Apache Airflow") + len("Apache Airflow"),
            },
        ]


@pytest.fixture(autouse=True)
def clear_ner_pipeline_cache() -> None:
    ner.get_ner_pipeline.cache_clear()


def test_ner_returns_normalized_entities_from_mocked_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pipeline = FakeNERPipeline()
    monkeypatch.setattr(ner, "pipeline", lambda *args, **kwargs: fake_pipeline)

    response = ner.extract_entities("Apache Airflow supports Python.")

    assert response.model == ner.NER_MODEL_NAME
    assert response.counts_by_type == {"MISC": 1, "ORG": 1}
    assert [entity.text for entity in response.entities] == ["Apache Airflow", "Python"]
    assert response.entities[0].type == "ORG"
    assert response.entities[0].confidence == 0.99
    assert response.entities[0].start == 0
    assert response.entities[0].end == len("Apache Airflow")


def test_ner_respects_max_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pipeline = FakeNERPipeline()
    monkeypatch.setattr(ner, "pipeline", lambda *args, **kwargs: fake_pipeline)

    response = ner.extract_entities("Apache Airflow supports Python.", max_entities=1)

    assert len(response.entities) == 1
    assert response.counts_by_type == {"ORG": 1}


def test_ner_pipeline_is_cached_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    created_pipelines: list[FakeNERPipeline] = []

    def fake_pipeline_factory(*args: object, **kwargs: object) -> FakeNERPipeline:
        pipeline = FakeNERPipeline()
        created_pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(ner, "pipeline", fake_pipeline_factory)

    ner.extract_entities("Apache Airflow supports Python.")
    ner.extract_entities("Apache Airflow supports Python.")

    assert len(created_pipelines) == 1
    assert created_pipelines[0].calls == 2


def test_ner_endpoint_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pipeline = FakeNERPipeline()
    monkeypatch.setattr(ner, "pipeline", lambda *args, **kwargs: fake_pipeline)
    client = TestClient(app)

    response = client.post(
        "/ner",
        json={"text": "Apache Airflow supports Python.", "max_entities": 100},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == ner.NER_MODEL_NAME
    assert body["counts_by_type"] == {"MISC": 1, "ORG": 1}
    assert body["entities"][0] == {
        "text": "Apache Airflow",
        "type": "ORG",
        "start": 0,
        "end": 14,
        "confidence": 0.99,
    }


def test_ner_endpoint_rejects_empty_text_with_validation_error() -> None:
    client = TestClient(app)

    response = client.post("/ner", json={"text": "   ", "max_entities": 100})

    assert response.status_code == 422


@pytest.mark.parametrize("max_entities", [0, 501])
def test_ner_endpoint_rejects_unsafe_max_entities(max_entities: int) -> None:
    client = TestClient(app)

    response = client.post(
        "/ner",
        json={"text": "Apache Airflow supports Python.", "max_entities": max_entities},
    )

    assert response.status_code == 422
