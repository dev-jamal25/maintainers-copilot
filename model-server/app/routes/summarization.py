from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.summarization import SummarizationRequest, SummarizationResponse
from app.services.summarization import summarize_text

logger = logging.getLogger(__name__)

router = APIRouter(tags=["summarization"])


@router.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest) -> SummarizationResponse:
    response = summarize_text(
        request.text,
        max_length=request.max_length,
        min_length=request.min_length,
    )
    logger.info(
        "summarization_completed",
        extra={
            "summary_length": len(response.summary),
            "text_length": len(request.text),
        },
    )
    return response
