from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import classify


class FakeClassifier:
    def __call__(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        # Deterministic: "bug" wins unless the text mentions docs.
        if "documentation" in text.lower():
            return {"bug": 0.1, "feature": 0.1, "docs": 0.7, "question": 0.1}
        return {"bug": 0.6, "feature": 0.2, "docs": 0.1, "question": 0.1}


@pytest.fixture(autouse=True)
def clear_classifier_cache() -> None:
    classify.get_classifier.cache_clear()


def test_classify_returns_argmax_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "build_classifier", FakeClassifier)
    response = classify.classify_issue("scheduler crashes", "it raises a traceback")
    assert response.label == "bug"
    assert response.model == classify.CLASSIFY_MODEL_NAME
    assert set(response.scores) == set(classify.CANDIDATE_LABELS)


def test_classify_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "build_classifier", FakeClassifier)
    response = classify.classify_issue("update the documentation", "the README is stale")
    assert response.label == "docs"


def test_classify_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "build_classifier", FakeClassifier)
    client = TestClient(app)
    response = client.post("/classify", json={"title": "scheduler crashes", "body": "traceback"})
    assert response.status_code == 200
    assert response.json()["label"] == "bug"


def test_classify_endpoint_rejects_blank_title() -> None:
    client = TestClient(app)
    assert client.post("/classify", json={"title": "   "}).status_code == 422
