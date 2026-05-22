"""Admin-invite business logic (service layer).

Owns the invite lifecycle and its transactions/audit. Routes call this; it speaks to repositories
and the fastapi-users ``UserManager`` (passed in via DI), never to HTTP or SQL directly. Every
invite create/accept writes a redacted audit row.
"""

from __future__ import annotations

import secrets as token_secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi_users import BaseUserManager
from fastapi_users import exceptions as fu_exceptions
from fastapi_users.schemas import BaseUserCreate

from app.core.config import Settings
from app.db.models import User
from app.domain.auth import Invitation, InviteAccept, InviteCreate
from app.domain.exceptions import ValidationDomainError
from app.infra.redaction import redact_value
from app.repositories.audit_repository import AuditRepository
from app.repositories.invitation_repository import InvitationRepository


class AuthInviteService:
    def __init__(
        self,
        invitations: InvitationRepository,
        audit: AuditRepository,
        settings: Settings,
    ) -> None:
        self._invitations = invitations
        self._audit = audit
        self._settings = settings

    async def create_invite(
        self, payload: InviteCreate, *, created_by: uuid.UUID, request_id: str | None
    ) -> Invitation:
        now = datetime.now(UTC)
        token = token_secrets.token_urlsafe(32)
        expires_at = now + timedelta(seconds=self._settings.auth.invite_lifetime_seconds)
        invite = await self._invitations.create(
            email=payload.email,
            role=payload.role,
            token=token,
            created_by=created_by,
            expires_at=expires_at,
            created_at=now,
        )
        await self._audit.record(
            action="invite.create",
            actor_id=created_by,
            target_type="invitation",
            target_id=str(invite.id),
            request_id=request_id,
            trace_id=None,
            details=redact_value({"email": payload.email, "role": payload.role.value}),
            created_at=now,
        )
        return invite

    async def accept_invite(
        self,
        payload: InviteAccept,
        *,
        user_manager: BaseUserManager[User, uuid.UUID],
        request_id: str | None,
    ) -> User:
        now = datetime.now(UTC)
        invite = await self._invitations.get_by_token(payload.token)
        if invite is None or not invite.is_usable(now=now):
            raise ValidationDomainError("Invalid or expired invitation.")

        try:
            user = await user_manager.create(
                BaseUserCreate(email=invite.email, password=payload.password), safe=True
            )
        except fu_exceptions.UserAlreadyExists as exc:
            raise ValidationDomainError("A user with this email already exists.") from exc
        except fu_exceptions.InvalidPasswordException as exc:
            raise ValidationDomainError("Password does not meet requirements.") from exc

        # Set the authoritative role; mirror admin onto is_superuser for fastapi-users compat.
        await user_manager.user_db.update(
            user, {"role": invite.role.value, "is_superuser": invite.role.value == "admin"}
        )
        await self._invitations.mark_used(invite.id, used_at=now)
        await self._audit.record(
            action="invite.accept",
            actor_id=user.id,
            target_type="user",
            target_id=str(user.id),
            request_id=request_id,
            trace_id=None,
            details=redact_value({"email": invite.email, "role": invite.role.value}),
            created_at=now,
        )
        return user
