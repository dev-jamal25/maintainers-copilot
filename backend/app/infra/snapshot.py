"""Retrieved-chunk snapshot seam (CLAUDE.md: chunk snapshots go to MinIO, after redaction).

This defines the port. The default ``NullSnapshotSink`` is a no-op so the RAG path runs without
object storage; a real MinIO-backed sink (``app/infra/minio_store.py``) is the documented follow-up.
Callers must pass already-redacted payloads.
"""

from __future__ import annotations

from typing import Protocol


class SnapshotSink(Protocol):
    async def write_snapshot(self, key: str, payload: dict[str, object]) -> None:
        """Persist a (redacted) snapshot under ``key``."""


class NullSnapshotSink:
    """No-op sink. Used until the MinIO adapter lands; keeps the RAG service decoupled."""

    async def write_snapshot(self, key: str, payload: dict[str, object]) -> None:
        return None
