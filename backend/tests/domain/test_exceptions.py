"""Domain exception hierarchy tests: stable codes, status mapping, user-safe messages."""

from __future__ import annotations

import pytest

from app.domain.exceptions import (
    AppError,
    AuthenticationError,
    ExternalServiceUnavailable,
    NotFoundError,
    PermissionDenied,
    ToolFailure,
    ValidationDomainError,
)

EXPECTED = [
    (NotFoundError, 404, "not_found"),
    (AuthenticationError, 401, "unauthenticated"),
    (PermissionDenied, 403, "permission_denied"),
    (ValidationDomainError, 422, "validation_error"),
    (ToolFailure, 502, "tool_failure"),
    (ExternalServiceUnavailable, 503, "service_unavailable"),
]


@pytest.mark.parametrize(("exc_type", "status", "code"), EXPECTED)
def test_status_and_code_mapping(exc_type: type[AppError], status: int, code: str) -> None:
    exc = exc_type()
    assert isinstance(exc, AppError)
    assert exc.status_code == status
    assert exc.code == code
    assert exc.message  # has a non-empty user-safe default message


def test_custom_message_and_internal_detail() -> None:
    exc = NotFoundError("conversation 7 not found", detail="user_id=42 mismatch")
    assert exc.message == "conversation 7 not found"
    assert exc.detail == "user_id=42 mismatch"
    # The internal detail must not be part of the user-facing string representation.
    assert "user_id=42" not in str(exc)


def test_default_message_used_when_none_given() -> None:
    assert PermissionDenied().message == PermissionDenied.default_message
