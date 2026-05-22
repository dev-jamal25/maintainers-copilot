"""RagIngestionService: children get embeddings, parents get NULL; JSONL loads + validates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from app.domain.rag import Chunk, ChunkLevel, ChunkMetadata, ChunkSourceType
from app.services.rag_ingestion import RagIngestionService, load_chunks_jsonl


def _chunk(chunk_id: str, level: ChunkLevel, parent_id: str | None) -> Chunk:
    return Chunk(
        level=level,
        text=f"text for {chunk_id}",
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            parent_id=parent_id,
            source_type=ChunkSourceType.DOCS,
            source_id="doc-x",
        ),
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.batches.append(list(texts))
        assert is_query is False  # ingestion embeds passages, not queries
        return [[float(len(text))] for text in texts]


class CapturingRepo:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, Sequence[float] | None]] = []

    async def insert_chunk(self, chunk: Chunk, embedding: Sequence[float] | None) -> None:
        self.inserts.append((chunk.chunk_id, embedding))


async def test_ingest_embeds_children_only() -> None:
    chunks = [
        _chunk("p0", ChunkLevel.PARENT, None),
        _chunk("p0:c0", ChunkLevel.CHILD, "p0"),
        _chunk("p0:c1", ChunkLevel.CHILD, "p0"),
    ]
    embedder = FakeEmbedder()
    repo = CapturingRepo()
    written = await RagIngestionService(embedder=embedder, repository=repo).ingest(chunks)

    assert written == 3
    by_id = dict(repo.inserts)
    assert by_id["p0"] is None  # parent stored without embedding
    assert by_id["p0:c0"] is not None and by_id["p0:c1"] is not None
    assert embedder.batches == [["text for p0:c0", "text for p0:c1"]]  # one batch, children only


async def test_ingest_respects_batch_size() -> None:
    children = [_chunk(f"c{i}", ChunkLevel.CHILD, "p0") for i in range(5)]
    embedder = FakeEmbedder()
    service = RagIngestionService(embedder=embedder, repository=CapturingRepo(), batch_size=2)
    await service.ingest(children)
    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]


def test_load_chunks_jsonl_parses_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [
        _chunk("p0", ChunkLevel.PARENT, None).model_dump(mode="json"),
        _chunk("p0:c0", ChunkLevel.CHILD, "p0").model_dump(mode="json"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = load_chunks_jsonl(path)
    assert [c.chunk_id for c in loaded] == ["p0", "p0:c0"]
    assert loaded[1].level is ChunkLevel.CHILD
