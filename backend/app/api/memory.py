"""Memory API: read-only inspector for the current user's episodic memories.

Writes happen only through the chatbot's ``write_memory`` tool (CLAUDE.md), so there is no write
route here. Reads are user-scoped via the authenticated user.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.auth import CurrentUserDep
from app.api.deps import ModelServerDep, SessionDep
from app.domain.memory import EpisodicMemory
from app.repositories.audit_repository import AuditRepository
from app.repositories.episodic_memory_repository import EpisodicMemoryRepository
from app.services.memory_service import MemoryService

router = APIRouter(tags=["memory"])


def get_memory_service(session: SessionDep, model_server: ModelServerDep) -> MemoryService:
    return MemoryService(
        memories=EpisodicMemoryRepository(session),
        embedder=model_server,
        audit=AuditRepository(session),
    )


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]


@router.get("/memories", response_model=list[EpisodicMemory])
async def list_memories(
    user: CurrentUserDep, service: MemoryServiceDep, limit: int = 50
) -> list[EpisodicMemory]:
    return await service.list_recent(user.id, limit=min(max(limit, 1), 200))
