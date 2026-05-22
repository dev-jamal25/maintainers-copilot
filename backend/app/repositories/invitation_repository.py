"""SQL-only repository for the ``invitations`` table (admin invite flow)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.auth import Invitation, UserRole

_INSERT_SQL = text(
    """
    INSERT INTO invitations (id, email, role, token, created_by, expires_at, used_at, created_at)
    VALUES (:id, :email, :role, :token, :created_by, :expires_at, NULL, :created_at)
    """
)

_SELECT_BY_TOKEN_SQL = text(
    """
    SELECT id, email, role, token, created_by, expires_at, used_at, created_at
    FROM invitations WHERE token = :token
    """
)

_MARK_USED_SQL = text(
    "UPDATE invitations SET used_at = :used_at WHERE id = :id AND used_at IS NULL"
)


class InvitationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        email: str,
        role: UserRole,
        token: str,
        created_by: UUID | None,
        expires_at: datetime,
        created_at: datetime,
    ) -> Invitation:
        invitation_id = uuid4()
        await self._session.execute(
            _INSERT_SQL,
            {
                "id": invitation_id,
                "email": email,
                "role": role.value,
                "token": token,
                "created_by": created_by,
                "expires_at": expires_at,
                "created_at": created_at,
            },
        )
        return Invitation(
            id=invitation_id,
            email=email,
            role=role,
            token=token,
            created_by=created_by,
            expires_at=expires_at,
            used_at=None,
            created_at=created_at,
        )

    async def get_by_token(self, token: str) -> Invitation | None:
        result = await self._session.execute(_SELECT_BY_TOKEN_SQL, {"token": token})
        row = result.mappings().first()
        if row is None:
            return None
        return Invitation(
            id=row["id"],
            email=row["email"],
            role=UserRole(row["role"]),
            token=row["token"],
            created_by=row["created_by"],
            expires_at=row["expires_at"],
            used_at=row["used_at"],
            created_at=row["created_at"],
        )

    async def mark_used(self, invitation_id: UUID, *, used_at: datetime) -> None:
        await self._session.execute(_MARK_USED_SQL, {"id": invitation_id, "used_at": used_at})
