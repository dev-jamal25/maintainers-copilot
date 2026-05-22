"""Redaction tests — the explicit CLAUDE.md proof that fake secrets never leak unredacted."""

from __future__ import annotations

from app.infra.redaction import REDACTED, redact, redact_value

# Fake, non-functional secrets used only to exercise the redactor (neutral variable names so
# ruff's hardcoded-secret rules don't flag the literals).
FAKE_OPENAI = "sk-ant-api03-AbCdEf0123456789AbCdEf0123456789"
FAKE_GH_CLASSIC = "ghp_AbCdEf0123456789AbCdEf0123456789abcd"
FAKE_GH_FINE = "github_pat_11ABCDE0aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMiJ9.s5ZBhEXAMPLEsignature_value"


def _assert_gone(secret: str, output: str) -> None:
    assert secret not in output, f"secret leaked: {output!r}"
    assert REDACTED in output


def test_openai_anthropic_key_redacted() -> None:
    out = redact(f"using key {FAKE_OPENAI} now")
    _assert_gone(FAKE_OPENAI, out)


def test_github_tokens_redacted() -> None:
    _assert_gone(FAKE_GH_CLASSIC, redact(f"token={FAKE_GH_CLASSIC}"))
    _assert_gone(FAKE_GH_FINE, redact(f"PAT is {FAKE_GH_FINE} ok"))


def test_bearer_token_redacted() -> None:
    # Bare bearer header (no "Authorization:" assignment prefix) exercises the Bearer rule directly.
    out = redact("Proxy header Bearer abc.def.ghi-token-value")
    assert "abc.def.ghi-token-value" not in out
    assert "Bearer [REDACTED]" in out


def test_jwt_redacted_standalone_and_in_header() -> None:
    _assert_gone(FAKE_JWT, redact(f"raw jwt {FAKE_JWT} end"))
    # Full Authorization header: the JWT must be gone (assignment + JWT rules both apply).
    _assert_gone(FAKE_JWT, redact(f"Authorization: Bearer {FAKE_JWT}"))


def test_assignment_keeps_key_drops_value() -> None:
    out = redact('db_password="hunter2-supersecret"')
    assert "hunter2-supersecret" not in out
    assert "db_password" in out
    assert REDACTED in out


def test_connection_string_credentials_redacted() -> None:
    out = redact("postgresql://admin:topsecretpw@db:5432/app")
    assert "topsecretpw" not in out
    assert "admin" not in out
    assert "db:5432/app" in out  # host/path preserved


def test_private_key_block_redacted() -> None:
    block = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Q\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact(f"key:\n{block}\nrest")
    assert "MIIBOgIBAAJB" not in out
    assert REDACTED in out


def test_email_pii_redacted() -> None:
    out = redact("contact alice.dev@example.com please")
    assert "alice.dev@example.com" not in out


def test_redact_value_recurses_and_uses_sensitive_keys() -> None:
    payload = {
        "note": f"key {FAKE_OPENAI}",
        "password": "shouldNotSurvive",
        "nested": {"authorization": "Bearer abc.def", "items": [FAKE_GH_CLASSIC, "safe text"]},
        "count": 7,
    }
    out = redact_value(payload)
    assert FAKE_OPENAI not in out["note"]
    assert out["password"] == REDACTED  # redacted by key name, regardless of value shape
    assert out["nested"]["authorization"] == REDACTED
    assert FAKE_GH_CLASSIC not in out["nested"]["items"][0]
    assert out["nested"]["items"][1] == "safe text"
    assert out["count"] == 7


def test_clean_text_unchanged() -> None:
    clean = "The scheduler restarts when the worker pod is OOMKilled."
    assert redact(clean) == clean
