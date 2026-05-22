"""RAG ingestion: load chunks, embed children via model-server, persist into ``rag_chunks``.

Separate from retrieval (CLAUDE.md: ingestion and retrieval are distinct flows). Child chunks are
embedded (they are what cosine search ranks); parent chunks are stored with a NULL embedding (they
exist only to expand context). The service is pure (takes a list of ``Chunk``) so it unit-tests with
a fake embedder + a capturing repository; ``run_file_ingestion`` is the runtime wiring (needs DB +
model-server) used by the ``python -m app.services.rag_ingestion`` entrypoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Protocol

from app.domain.rag import Chunk, ChunkLevel

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class EmbedderPort(Protocol):
    async def embed(self, texts: Sequence[str], *, is_query: bool = ...) -> list[list[float]]: ...


class ChunkWriterPort(Protocol):
    async def insert_chunk(self, chunk: Chunk, embedding: Sequence[float] | None) -> None: ...


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    """Parse a chunks JSONL file (the ml/eval handoff format) into validated domain chunks."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                chunks.append(Chunk.model_validate(json.loads(stripped)))
    return chunks


class RagIngestionService:
    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        repository: ChunkWriterPort,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._embedder = embedder
        self._repository = repository
        self._batch_size = batch_size

    async def ingest(self, chunks: list[Chunk]) -> int:
        """Embed child chunks (batched) and persist every chunk. Returns the count written."""
        children = [chunk for chunk in chunks if chunk.level == ChunkLevel.CHILD]
        embeddings: dict[str, list[float]] = {}
        for batch in batched(children, self._batch_size):
            vectors = await self._embedder.embed([c.text for c in batch], is_query=False)
            for chunk, vector in zip(batch, vectors, strict=True):
                embeddings[chunk.chunk_id] = vector

        for chunk in chunks:
            await self._repository.insert_chunk(chunk, embeddings.get(chunk.chunk_id))
        logger.info(
            "rag_ingestion_completed",
            extra={"total": len(chunks), "embedded": len(embeddings)},
        )
        return len(chunks)


async def run_file_ingestion(path: Path) -> int:  # pragma: no cover - runtime wiring (needs DB)
    """Wire real Settings/engine/model-server, ingest a JSONL file, commit. Not unit-tested."""
    from app.core.config import Settings
    from app.db.session import create_engine, create_session_factory
    from app.infra.model_server_client import ModelServerClient
    from app.repositories.chunk_repository import ChunkRepository

    settings = Settings()
    engine = create_engine(settings.database)
    session_factory = create_session_factory(engine)
    model_server = ModelServerClient(settings.model_server)
    try:
        chunks = load_chunks_jsonl(path)
        async with session_factory() as session:
            service = RagIngestionService(
                embedder=model_server, repository=ChunkRepository(session)
            )
            written = await service.ingest(chunks)
            await session.commit()
        return written
    finally:
        await model_server.aclose()
        await engine.dispose()


def main() -> int:  # pragma: no cover - CLI entrypoint
    import sys

    if len(sys.argv) != 2:
        logger.error("usage: python -m app.services.rag_ingestion <chunks.jsonl>")
        return 2
    written = asyncio.run(run_file_ingestion(Path(sys.argv[1])))
    logger.info("ingested_chunks", extra={"written": written})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
