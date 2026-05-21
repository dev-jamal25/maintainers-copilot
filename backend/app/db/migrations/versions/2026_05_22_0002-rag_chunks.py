"""rag chunks table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-22

Creates the ``rag_chunks`` table that persists the chosen RAG index into Postgres + pgvector
(the frozen vector store, DECISIONS D3). Columns mirror the canonical chunk contract
(``backend/app/domain/rag.py``): the D3.6 metadata fields, the chunk text, and a 384-dim
``vector`` embedding. 384 is stable across the bge-small / all-MiniLM comparison (both 384-dim),
so the column dimension does not change with the embedding choice.

Defined as raw SQL (no SQLAlchemy model) to keep ``Base.metadata`` free of feature tables for
now; the SQL-only ChunkRepository reads/writes this table.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE rag_chunks (
            chunk_id          TEXT PRIMARY KEY,
            parent_id         TEXT,
            level             TEXT NOT NULL,
            source_type       TEXT NOT NULL,
            source_id         TEXT NOT NULL,
            title             TEXT,
            url               TEXT,
            section_path      JSONB NOT NULL DEFAULT '[]'::jsonb,
            airflow_area      TEXT,
            tags              JSONB NOT NULL DEFAULT '[]'::jsonb,
            version           TEXT,
            github_issue_id   BIGINT,
            github_comment_id BIGINT,
            created_at        TIMESTAMPTZ,
            closed_at         TIMESTAMPTZ,
            text              TEXT NOT NULL,
            embedding         vector(384)
        )
        """
    )
    op.execute("CREATE INDEX ix_rag_chunks_level ON rag_chunks (level)")
    op.execute("CREATE INDEX ix_rag_chunks_source ON rag_chunks (source_type, source_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rag_chunks")
