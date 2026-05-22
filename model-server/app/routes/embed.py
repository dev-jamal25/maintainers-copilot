from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services.embed import embed_texts

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    response = embed_texts(request.texts, is_query=request.is_query)
    logger.info(
        "embed_completed",
        extra={"count": len(request.texts), "is_query": request.is_query, "dim": response.dim},
    )
    return response
