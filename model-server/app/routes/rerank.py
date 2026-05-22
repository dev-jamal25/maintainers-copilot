from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.rerank import RerankRequest, RerankResponse
from app.services.rerank import rerank_passages

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rerank"])


@router.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    response = rerank_passages(request.query, request.passages, top_k=request.top_k)
    logger.info(
        "rerank_completed",
        extra={"passages": len(request.passages), "returned": len(response.results)},
    )
    return response
