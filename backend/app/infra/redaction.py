"""Secret + PII redaction (CLAUDE.md observability requirement).

Runs before anything leaves the process: structured logs, Langfuse traces, long-term memory
writes, and retrieved-chunk snapshots. Pure-stdlib (regex only) so it stays importable from every
layer with no heavy dependency. The module-level ``redact`` (strings) and ``redact_value``
(arbitrary JSON-ish values: dict/list/scalars) are the app-facing surface; callers should never
log/trace/persist raw I/O without passing it through one of these first.

Covered (CLAUDE.md): OpenAI/Anthropic-style keys (``sk-…``), GitHub tokens (``ghp_…`` /
``github_pat_…`` and the gho_/ghu_/ghr_/ghs_ family), bearer + JWT tokens, ``password``/``secret``/
``token`` assignments, PEM private-key blocks, connection-string credentials, and email PII. The
redaction is intentionally aggressive (fail-safe): over-redaction is acceptable, leaking is not.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

# PEM private-key blocks (RSA/EC/OPENSSH/etc.), including the body between the markers.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)

# key=value / key: value assignments for sensitive keys (value redacted, key kept for debugging).
_ASSIGNMENT = re.compile(
    r"(?P<key>(?i:password|passwd|secret|api[_-]?key|access[_-]?key"
    r"|secret[_-]?key|token|authorization))"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<quote>[\"']?)[^\s\"',;]+(?P=quote)"
)

# Authorization: Bearer <token>
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)

# JWTs: three base64url segments, header starts with the canonical {"alg":... -> "eyJ".
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")

# GitHub fine-grained PAT and classic/oauth/app tokens.
_GITHUB_PAT = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")

# OpenAI / Anthropic style API keys (sk- and sk-ant-).
_API_KEY = re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}\b")

# Credentials embedded in a connection string: scheme://user:pass@host -> redact user:pass.
_CONN_CREDS = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)[^/\s:@]+:[^/\s:@]+@")

# Email PII.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def _redact_assignment(match: re.Match[str]) -> str:
    return f"{match.group('key')}{match.group('sep')}{REDACTED}"


def _redact_conn(match: re.Match[str]) -> str:
    return f"{match.group('scheme')}{REDACTED}:{REDACTED}@"


def redact(text: str) -> str:
    """Return ``text`` with secrets and PII replaced by ``[REDACTED]``.

    Order matters: structural patterns (private keys, assignments, connection creds) run before
    token-shape patterns so a redacted value is never partially re-matched.
    """
    if not text:
        return text
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _ASSIGNMENT.sub(_redact_assignment, text)
    text = _CONN_CREDS.sub(_redact_conn, text)
    text = _BEARER.sub(f"Bearer {REDACTED}", text)
    text = _JWT.sub(REDACTED, text)
    text = _GITHUB_PAT.sub(REDACTED, text)
    text = _GITHUB_TOKEN.sub(REDACTED, text)
    text = _API_KEY.sub(REDACTED, text)
    text = _EMAIL.sub(REDACTED, text)
    return text


_SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "access_key",
)


def _is_sensitive_key(key: object) -> bool:
    return isinstance(key, str) and any(part in key.lower() for part in _SENSITIVE_KEY_SUBSTRINGS)


def redact_value(value: Any) -> Any:
    """Recursively redact a JSON-ish value.

    Strings are passed through :func:`redact`. In mappings, a value under a sensitive-looking key
    (``password``, ``token``, …) is fully replaced regardless of its shape; other values recurse.
    Lists/tuples recurse element-wise. Other scalars are returned unchanged.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {
            key: (REDACTED if _is_sensitive_key(key) else redact_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
