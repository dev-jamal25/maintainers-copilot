"""Domain exception hierarchy (CLAUDE.md error model).

Business logic raises these; a single API exception handler maps them to structured HTTP
responses. Each carries a stable ``code`` (machine-readable, safe to expose), a ``status_code``,
and a user-safe ``message``. The optional ``detail`` is internal-only diagnostic context that the
API handler must never echo to clients (it goes to logs/traces — after redaction).

Infra adapters raise their own typed errors (see ``app/infra/errors.py``); services catch those and
convert them into the appropriate domain exception so the API layer only ever sees this hierarchy.
"""

from __future__ import annotations


class AppError(Exception):
    """Base domain error. Subclasses set ``status_code``/``code``/``default_message``."""

    status_code: int = 500
    code: str = "app_error"
    default_message: str = "An unexpected application error occurred."

    def __init__(self, message: str | None = None, *, detail: str | None = None) -> None:
        self.message = message if message is not None else self.default_message
        self.detail = detail
        super().__init__(self.message)


class NotFoundError(AppError):
    """A requested resource does not exist or is not visible to the caller."""

    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."


class AuthenticationError(AppError):
    """No valid credentials were supplied (401)."""

    status_code = 401
    code = "unauthenticated"
    default_message = "Authentication is required."


class PermissionDenied(AppError):
    """Authenticated but not allowed to perform the action (403)."""

    status_code = 403
    code = "permission_denied"
    default_message = "You do not have permission to perform this action."


class ValidationDomainError(AppError):
    """A domain invariant or input constraint was violated (422)."""

    status_code = 422
    code = "validation_error"
    default_message = "The request was invalid."


class ToolFailure(AppError):
    """A chatbot tool failed. Usually surfaced to the LLM as a structured tool error rather than
    raised to the API; when it does propagate it maps to a bad-gateway response."""

    status_code = 502
    code = "tool_failure"
    default_message = "A tool failed to complete."


class ExternalServiceUnavailable(AppError):
    """A required downstream (model-server, LLM, Redis, MinIO, DB) is unavailable (503)."""

    status_code = 503
    code = "service_unavailable"
    default_message = "A required service is currently unavailable."
