from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Final, Protocol, cast

from transformers import pipeline

from app.schemas.ner import Entity, NERResponse

NER_MODEL_NAME: Final[str] = "dslim/distilbert-NER"
MAX_NER_INPUT_CHARS: Final[int] = 20_000


class PipelineCallable(Protocol):
    def __call__(self, text: str) -> list[Mapping[str, Any]]:
        """Run token classification for one text payload."""


def pipeline_device() -> int:
    try:
        import torch
    except ImportError:
        return -1
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def get_ner_pipeline() -> PipelineCallable:
    return cast(
        PipelineCallable,
        pipeline(
            "token-classification",
            model=NER_MODEL_NAME,
            aggregation_strategy="simple",
            device=pipeline_device(),
        ),
    )


def normalize_entity(raw_entity: Mapping[str, Any], source_text: str) -> Entity | None:
    entity_type = raw_entity.get("entity_group") or raw_entity.get("entity")
    start = raw_entity.get("start")
    end = raw_entity.get("end")

    if entity_type is None or start is None or end is None:
        return None

    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None

    if start_int < 0 or end_int <= start_int or end_int > len(source_text):
        return None

    text = source_text[start_int:end_int].strip()
    if not text:
        text = str(raw_entity.get("word", "")).strip()
    if not text:
        return None

    confidence = raw_entity.get("score", 0.0)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.0

    normalized_type = str(entity_type).removeprefix("B-").removeprefix("I-")
    return Entity(
        confidence=max(0.0, min(1.0, confidence_float)),
        end=end_int,
        start=start_int,
        text=text,
        type=normalized_type,
    )


def dedupe_entities(entities: list[Entity]) -> list[Entity]:
    seen: set[tuple[str, str, int, int]] = set()
    deduped: list[Entity] = []
    for entity in entities:
        key = (entity.text, entity.type, entity.start, entity.end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def extract_entities(text: str, *, max_entities: int = 100) -> NERResponse:
    source_text = text[:MAX_NER_INPUT_CHARS]
    raw_entities = get_ner_pipeline()(source_text)
    normalized_entities = [
        entity
        for raw_entity in raw_entities
        if (entity := normalize_entity(raw_entity, source_text)) is not None
    ]
    entities = sorted(
        dedupe_entities(normalized_entities), key=lambda item: (item.start, item.type)
    )
    capped_entities = entities[:max_entities]
    counts = Counter(entity.type for entity in capped_entities)
    return NERResponse(
        counts_by_type=dict(sorted(counts.items())),
        entities=capped_entities,
        model=NER_MODEL_NAME,
    )
