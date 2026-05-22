from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.classify import router as classify_router
from app.routes.embed import router as embed_router
from app.routes.ner import router as ner_router
from app.routes.rerank import router as rerank_router
from app.routes.summarization import router as summarization_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Maintainer's Copilot Model Server")
    app.include_router(ner_router)
    app.include_router(summarization_router)
    app.include_router(embed_router)
    app.include_router(rerank_router)
    app.include_router(classify_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "model_server_unhandled_exception",
            extra={"method": request.method, "path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
