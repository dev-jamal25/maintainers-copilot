"""SQL-only repository for the ``rag_chunks`` table (pgvector store).

Owns the SQL for persisting the chosen RAG index and running cosine retrieval against
pgvector (frozen vector store, DECISIONS D3). No HTTP concerns, no external service calls —
services orchestrate transactions and the retrieval pipeline; this layer only speaks SQL.

Embeddings are bound using pgvector's text format (``[0.1,0.2,...]``) cast to ``vector``, so no
extra Python dependency is required. Cosine similarity uses pgvector's ``<=>`` operator.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.rag import Chunk, RetrievedChunk


@dataclass(frozen=True)
class StoredChunk:
    """A persisted chunk's text + the metadata needed to build context and citations."""

    chunk_id: str
    parent_id: str | None
    level: str
    source_type: str
    title: str | None
    url: str | None
    text: str


_INSERT_SQL = text(
    """
    INSERT INTO rag_chunks (
        chunk_id, parent_id, level, source_type, source_id, title, url, section_path,
        airflow_area, tags, version, github_issue_id, github_comment_id, created_at,
        closed_at, text, embedding
    ) VALUES (
        :chunk_id, :parent_id, :level, :source_type, :source_id, :title, :url,
        CAST(:section_path AS jsonb), :airflow_area, CAST(:tags AS jsonb), :version,
        :github_issue_id, :github_comment_id, :created_at, :closed_at, :text,
        CAST(:embedding AS vector)
    )
    ON CONFLICT (chunk_id) DO UPDATE SET
        text = EXCLUDED.text,
        embedding = EXCLUDED.embedding
    """
)

_SEARCH_SQL = text(
    """
    SELECT chunk_id, parent_id, (1 - (embedding <=> CAST(:query AS vector))) AS score
    FROM rag_chunks
    WHERE level = :level AND embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :limit
    """
)

# Same as _SEARCH_SQL plus a metadata filter on source_type (D-RAG metadata filtering).
_SEARCH_SQL_SOURCE = text(
    """
    SELECT chunk_id, parent_id, (1 - (embedding <=> CAST(:query AS vector))) AS score
    FROM rag_chunks
    WHERE level = :level AND embedding IS NOT NULL AND source_type = :source_type
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :limit
    """
)

_FETCH_BY_IDS_SQL = text(
    """
    SELECT chunk_id, parent_id, level, source_type, title, url, text
    FROM rag_chunks
    WHERE chunk_id = ANY(:ids)
    """
)


def encode_vector(vector: Sequence[float]) -> str:
    """Encode a float vector in pgvector's text format: ``[v0,v1,...]``."""
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


class ChunkRepository:
    """Persists chunks and runs cosine retrieval against ``rag_chunks`` (pgvector)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_chunk(self, chunk: Chunk, embedding: Sequence[float] | None) -> None:
        metadata = chunk.metadata
        await self._session.execute(
            _INSERT_SQL,
            {
                "chunk_id": metadata.chunk_id,
                "parent_id": metadata.parent_id,
                "level": chunk.level.value,
                "source_type": metadata.source_type.value,
                "source_id": metadata.source_id,
                "title": metadata.title,
                "url": metadata.url,
                "section_path": json.dumps(metadata.section_path),
                "airflow_area": metadata.airflow_area,
                "tags": json.dumps(metadata.tags),
                "version": metadata.version,
                "github_issue_id": metadata.github_issue_id,
                "github_comment_id": metadata.github_comment_id,
                "created_at": metadata.created_at,
                "closed_at": metadata.closed_at,
                "text": chunk.text,
                "embedding": encode_vector(embedding) if embedding is not None else None,
            },
        )

    async def search_cosine(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        level: str = "child",
        source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        params: dict[str, object] = {
            "query": encode_vector(query_embedding),
            "level": level,
            "limit": limit,
        }
        if source_type is None:
            statement = _SEARCH_SQL
        else:
            statement = _SEARCH_SQL_SOURCE
            params["source_type"] = source_type
        result = await self._session.execute(statement, params)
        return [
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                parent_id=row["parent_id"],
                score=float(row["score"]),
                rank=rank,
            )
            for rank, row in enumerate(result.mappings().all())
        ]

    async def fetch_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]:
        """Return text + metadata for the given chunk ids (for context expansion + citations)."""
        if not chunk_ids:
            return {}
        result = await self._session.execute(_FETCH_BY_IDS_SQL, {"ids": list(chunk_ids)})
        stored: dict[str, StoredChunk] = {}
        for row in result.mappings().all():
            chunk_id = str(row["chunk_id"])
            stored[chunk_id] = StoredChunk(
                chunk_id=chunk_id,
                parent_id=row["parent_id"],
                level=str(row["level"]),
                source_type=str(row["source_type"]),
                title=row["title"],
                url=row["url"],
                text=str(row["text"]),
            )
        return stored
