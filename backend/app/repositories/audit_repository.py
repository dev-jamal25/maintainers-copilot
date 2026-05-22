"""SQL-only repository for the ``audit_log`` table.

One row per sensitive action (CLAUDE.md): actor, action, target, request/trace id, and a redacted
``details`` JSON. Callers (services) must redact ``details`` before passing it in — the repository
only speaks SQL.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_INSERT_SQL = text(
    """
    INSERT INTO audit_log (
        id, actor_id, action, target_type, target_id, request_id, trace_id, details, created_at
    ) VALUES (
        :id, :actor_id, :action, :target_type, :target_id, :request_id, :trace_id,
        CAST(:details AS jsonb), :created_at
    )
    """
)


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        action: str,
        actor_id: UUID | None,
        target_type: str | None,
        target_id: str | None,
        request_id: str | None,
        trace_id: str | None,
        details: dict[str, object] | None,
        created_at: datetime,
    ) -> UUID:
        audit_id = uuid4()
        await self._session.execute(
            _INSERT_SQL,
            {
                "id": audit_id,
                "actor_id": actor_id,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "request_id": request_id,
                "trace_id": trace_id,
                "details": json.dumps(details or {}),
                "created_at": created_at,
            },
        )
        return audit_id
