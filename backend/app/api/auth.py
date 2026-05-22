"""Auth wiring (fastapi-users 15.x) + admin invite routes.

- JWT bearer login at ``/auth/jwt/login`` (signing secret from Vault, per-request strategy).
- Roles are ``user``/``admin``; ``require_admin`` returns 403 (``PermissionDenied``) for non-admins,
  while missing/invalid credentials yield 401 from fastapi-users.
- Registration is invite-only: an admin mints an invite, the recipient redeems it to set a password.
  No password reset / email verification routers are mounted (out of scope).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from pydantic import BaseModel, ConfigDict, EmailStr

from app.api.deps import RequestIdDep, SessionDep, get_settings
from app.core.config import Settings
from app.db.models import User
from app.domain.auth import InviteAccept, InviteCreate, UserRole
from app.domain.exceptions import PermissionDenied
from app.infra.secrets import load_jwt_secret
from app.repositories.audit_repository import AuditRepository
from app.repositories.invitation_repository import InvitationRepository
from app.services.auth_service import AuthInviteService

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


# --- schemas -----------------------------------------------------------------


class UserRead(schemas.BaseUser[uuid.UUID]):
    role: str


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


class InviteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    email: EmailStr
    role: UserRole
    expires_at: datetime


# --- DI: user db, jwt secret, manager, strategy ------------------------------


async def get_user_db(
    session: SessionDep,
) -> AsyncIterator[SQLAlchemyUserDatabase[User, uuid.UUID]]:
    yield SQLAlchemyUserDatabase(session, User)


def get_jwt_secret(settings: Annotated[Settings, Depends(get_settings)]) -> str:
    return load_jwt_secret(settings.vault)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """User manager. Reset/verification token secrets are set but their routers are not mounted."""

    def __init__(self, user_db: SQLAlchemyUserDatabase[User, uuid.UUID], secret: str) -> None:
        super().__init__(user_db)
        self.reset_password_token_secret = secret
        self.verification_token_secret = secret


async def get_user_manager(
    user_db: Annotated[SQLAlchemyUserDatabase[User, uuid.UUID], Depends(get_user_db)],
    secret: Annotated[str, Depends(get_jwt_secret)],
) -> AsyncIterator[UserManager]:
    yield UserManager(user_db, secret)


def get_jwt_strategy(
    secret: Annotated[str, Depends(get_jwt_secret)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> JWTStrategy[User, uuid.UUID]:
    return JWTStrategy(secret=secret, lifetime_seconds=settings.auth.jwt_lifetime_seconds)


auth_backend = AuthenticationBackend(
    name="jwt", transport=bearer_transport, get_strategy=get_jwt_strategy
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)


async def require_admin(user: Annotated[User, Depends(current_active_user)]) -> User:
    """Authenticated admins only. Non-admins -> 403; missing credentials -> 401 (upstream)."""
    if user.role != UserRole.ADMIN.value:
        raise PermissionDenied("Admin role is required for this action.")
    return user


CurrentUserDep = Annotated[User, Depends(current_active_user)]
AdminDep = Annotated[User, Depends(require_admin)]


# --- routers -----------------------------------------------------------------

router = APIRouter()
router.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"]
)

invite_router = APIRouter(prefix="/auth/invitations", tags=["auth"])


def _invite_service(
    session: SessionDep, settings: Annotated[Settings, Depends(get_settings)]
) -> AuthInviteService:
    return AuthInviteService(InvitationRepository(session), AuditRepository(session), settings)


InviteServiceDep = Annotated[AuthInviteService, Depends(_invite_service)]


@invite_router.post("", response_model=InviteResponse, status_code=201)
async def create_invitation(
    payload: InviteCreate,
    admin: AdminDep,
    service: InviteServiceDep,
    session: SessionDep,
    request_id: RequestIdDep,
) -> InviteResponse:
    invite = await service.create_invite(payload, created_by=admin.id, request_id=request_id)
    await session.commit()
    return InviteResponse(
        token=invite.token, email=invite.email, role=invite.role, expires_at=invite.expires_at
    )


@invite_router.post("/accept", response_model=UserRead)
async def accept_invitation(
    payload: InviteAccept,
    service: InviteServiceDep,
    user_manager: Annotated[UserManager, Depends(get_user_manager)],
    session: SessionDep,
    request_id: RequestIdDep,
) -> User:
    user = await service.accept_invite(payload, user_manager=user_manager, request_id=request_id)
    await session.commit()
    return user


router.include_router(invite_router)
