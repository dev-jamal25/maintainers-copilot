"""SQLAlchemy ORM models.

Only the ``User`` table is mapped as an ORM model, because fastapi-users-db-sqlalchemy operates on
an ORM model. The other Day-4 tables (conversations, messages, widgets, invitations, audit_log) and
the vector table (episodic_memories, which carries a pgvector ``vector(384)`` column) are created by
the hand-written Alembic migration and read/written by raw-SQL repositories — consistent with the
existing ``rag_chunks`` decision to keep ``Base.metadata`` free of the pgvector type dependency.

``User`` adds a ``role`` column (``user``/``admin``) — the authoritative authorization field — and a
``created_at`` timestamp on top of the fastapi-users base columns.
"""

from __future__ import annotations

from datetime import datetime

from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from sqlalchemy import String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.domain.auth import UserRole


class User(SQLAlchemyBaseUserTableUUID, Base):
    """User table: fastapi-users base columns + role + created_at.

    Base columns inherited: id (UUID PK), email (unique, indexed), hashed_password, is_active,
    is_superuser, is_verified.
    """

    # fastapi-users defaults to the singular "user"; pin to "users" to match migration 0003 + repos.
    __tablename__ = "users"

    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default=UserRole.USER.value, server_default=UserRole.USER.value
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
