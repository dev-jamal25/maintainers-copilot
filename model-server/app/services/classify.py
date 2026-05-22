"""Issue classification service.

Runtime endpoint over the four frozen labels (bug/feature/docs/question). The deployed default here
is a zero-shot NLI classifier (no fine-tune weights to ship in this container); the fine-tuned
DistilBERT remains the DECISIONS deployment candidate and can replace ``build_classifier`` later.
Lazy-loaded + cached; tests monkeypatch ``build_classifier`` so CI runs without downloads.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final, Protocol

from app.schemas.classify import ClassifyResponse

CLASSIFY_MODEL_NAME: Final[str] = "facebook/bart-large-mnli"
CANDIDATE_LABELS: Final[tuple[str, ...]] = ("bug", "feature", "docs", "question")
MAX_INPUT_CHARS: Final[int] = 8_000


class ClassifierCallable(Protocol):
    def __call__(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """Return a label -> score mapping over the candidate labels."""


class ZeroShotClassifier:
    def __init__(self) -> None:
        from transformers import pipeline

        self._pipeline = pipeline("zero-shot-classification", model=CLASSIFY_MODEL_NAME)

    def __call__(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        output = self._pipeline(text, candidate_labels=list(labels))
        return {
            str(label): float(score)
            for label, score in zip(output["labels"], output["scores"], strict=True)
        }


def build_classifier() -> ClassifierCallable:
    return ZeroShotClassifier()


@lru_cache(maxsize=1)
def get_classifier() -> ClassifierCallable:
    return build_classifier()


def classify_issue(title: str, body: str = "") -> ClassifyResponse:
    text = f"{title}\n\n{body}".strip()[:MAX_INPUT_CHARS]
    scores = get_classifier()(text, CANDIDATE_LABELS)
    label = max(scores, key=lambda key: scores[key]) if scores else CANDIDATE_LABELS[0]
    return ClassifyResponse(label=label, scores=scores, model=CLASSIFY_MODEL_NAME)
