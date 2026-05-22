from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.classify import ClassifyRequest, ClassifyResponse
from app.services.classify import classify_issue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    response = classify_issue(request.title, request.body)
    logger.info("classify_completed", extra={"label": response.label})
    return response
