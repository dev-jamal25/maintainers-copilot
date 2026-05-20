from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.ner import NERRequest, NERResponse
from app.services.ner import extract_entities

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ner"])


@router.post("/ner", response_model=NERResponse)
def extract_ner(request: NERRequest) -> NERResponse:
    response = extract_entities(request.text, max_entities=request.max_entities)
    logger.info(
        "ner_extraction_completed",
        extra={
            "entity_count": len(response.entities),
            "entity_types": sorted(response.counts_by_type),
            "text_length": len(request.text),
        },
    )
    return response
