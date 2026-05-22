"""Auth domain models: roles and the admin-invite contract.

Roles are ``user`` / ``admin`` (CLAUDE.md). Authorization is role-based: a custom dependency checks
``role == admin`` and raises ``PermissionDenied`` (403); auth failures are 401. No password reset,
no email verification (deliberately out of scope for the deadline).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class Invitation(BaseModel):
    """A pending or consumed admin invite."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    email: EmailStr
    role: UserRole
    token: str
    created_by: UUID | None = None
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime

    def is_usable(self, *, now: datetime) -> bool:
        """True when the invite has neither been consumed nor expired."""
        return self.used_at is None and now < self.expires_at


class InviteCreate(BaseModel):
    """Admin payload to mint an invite."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    role: UserRole = UserRole.USER


class InviteAccept(BaseModel):
    """Public payload to redeem an invite token and set a password."""

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=8)
    password: str = Field(min_length=8, max_length=128)
