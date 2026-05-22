"""Thin synchronous backend API client for the Streamlit internal app.

Streamlit talks to the FastAPI backend over HTTP only (no DB access). Pure ``httpx`` so it has no
Streamlit dependency and is unit-testable on its own.
"""

from __future__ import annotations

from typing import Any

import httpx

_TIMEOUT = 60.0


class ApiError(Exception):
    """A user-presentable backend error (status + best-effort message)."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"{status_code}: {message}")


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    message = response.text
    try:
        body = response.json()
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            message = str(body["error"].get("message", message))
    except ValueError:
        pass
    raise ApiError(response.status_code, message)


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")

    def _auth(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def login(self, email: str, password: str) -> str:
        # fastapi-users JWT login expects form fields username/password.
        response = httpx.post(
            f"{self._base}/auth/jwt/login",
            data={"username": email, "password": password},
            timeout=_TIMEOUT,
        )
        _raise_for_status(response)
        return str(response.json()["access_token"])

    def me(self, token: str) -> dict[str, Any]:
        response = httpx.get(f"{self._base}/users/me", headers=self._auth(token), timeout=_TIMEOUT)
        _raise_for_status(response)
        return dict(response.json())

    def chat(
        self, token: str, message: str, conversation_id: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        response = httpx.post(
            f"{self._base}/chat", json=payload, headers=self._auth(token), timeout=_TIMEOUT
        )
        _raise_for_status(response)
        return dict(response.json())

    def list_memories(self, token: str) -> list[dict[str, Any]]:
        response = httpx.get(f"{self._base}/memories", headers=self._auth(token), timeout=_TIMEOUT)
        _raise_for_status(response)
        return list(response.json())

    def list_widgets(self, token: str) -> list[dict[str, Any]]:
        response = httpx.get(f"{self._base}/widgets", headers=self._auth(token), timeout=_TIMEOUT)
        _raise_for_status(response)
        return list(response.json())

    def create_widget(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self._base}/widgets", json=payload, headers=self._auth(token), timeout=_TIMEOUT
        )
        _raise_for_status(response)
        return dict(response.json())

    def create_invite(self, token: str, email: str, role: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self._base}/auth/invitations",
            json={"email": email, "role": role},
            headers=self._auth(token),
            timeout=_TIMEOUT,
        )
        _raise_for_status(response)
        return dict(response.json())
