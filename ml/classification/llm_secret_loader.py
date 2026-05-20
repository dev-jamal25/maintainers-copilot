from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import hvac

VAULT_ADDR_ENV = "VAULT_ADDR"
VAULT_TOKEN_ENV = "VAULT_TOKEN"  # noqa: S105 - environment variable name, not a token value.
VAULT_DEV_TOKEN_ENV = "VAULT_DEV_ROOT_TOKEN_ID"  # noqa: S105 - variable name only.
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
GENERIC_LLM_API_KEY_ENV = "LLM_API_KEY"

VAULT_MOUNT_POINT = "secret"
VAULT_SECRET_PATH = "maintainers-copilot/anthropic"  # noqa: S105 - Vault path only.
VAULT_SECRET_FIELD = "api_key"  # noqa: S105 - Vault field name only.

SETUP_INSTRUCTIONS = (
    "Set an Anthropic API key with one safe local option:\n"
    "PowerShell:\n"
    '$env:VAULT_ADDR = "http://localhost:8200"\n'
    '$env:VAULT_TOKEN = "dev-only-root-token"\n\n'
    "Vault CLI through Docker:\n"
    "docker compose exec -e VAULT_ADDR=http://127.0.0.1:8200 "
    "-e VAULT_TOKEN=dev-only-root-token vault vault kv put "
    'secret/maintainers-copilot/anthropic api_key="$env:ANTHROPIC_API_KEY"\n\n'
    "Environment fallback:\n"
    '$env:ANTHROPIC_API_KEY = "<your local key>"\n'
)


class LLMSecretError(RuntimeError):
    """Raised when no usable LLM API key can be loaded."""


class VaultSecretError(RuntimeError):
    """Raised when Vault cannot return a usable LLM API key."""


@dataclass(frozen=True)
class SecretLookupResult:
    api_key: str = field(repr=False)
    source: str
    source_detail: str


ClientFactory = Callable[..., Any]


def is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    placeholder_markers = (
        "changeme",
        "example",
        "fake",
        "placeholder",
        "replace-me",
        "your-",
    )
    return any(marker in normalized for marker in placeholder_markers)


def vault_token_from_env(env: Mapping[str, str]) -> str:
    return env.get(VAULT_TOKEN_ENV, "").strip() or env.get(VAULT_DEV_TOKEN_ENV, "").strip()


def load_from_vault(
    *,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory = hvac.Client,
    timeout_seconds: float = 10.0,
) -> SecretLookupResult:
    resolved_env = os.environ if env is None else env
    vault_addr = resolved_env.get(VAULT_ADDR_ENV, "").strip()
    vault_token = vault_token_from_env(resolved_env)
    if not vault_addr:
        raise VaultSecretError(f"{VAULT_ADDR_ENV} is not set")
    if not vault_token:
        raise VaultSecretError(f"{VAULT_TOKEN_ENV} is not set")

    try:
        client = client_factory(url=vault_addr, token=vault_token, timeout=timeout_seconds)
        seal_status = client.sys.read_seal_status()
        if seal_status.get("sealed"):
            raise VaultSecretError("Vault is sealed")
        if not seal_status.get("initialized", True):
            raise VaultSecretError("Vault is not initialized")
        if not client.is_authenticated():
            raise VaultSecretError("Vault token is invalid")
        response = client.secrets.kv.v2.read_secret_version(
            mount_point=VAULT_MOUNT_POINT,
            path=VAULT_SECRET_PATH,
        )
    except VaultSecretError:
        raise
    except Exception as exc:
        raise VaultSecretError(f"Vault lookup failed for {VAULT_SECRET_PATH}") from exc

    data = response.get("data", {}).get("data", {})
    if not isinstance(data, dict):
        raise VaultSecretError(f"Vault path {VAULT_SECRET_PATH} returned no data")
    raw_value = data.get(VAULT_SECRET_FIELD)
    if not isinstance(raw_value, str) or is_placeholder_secret(raw_value):
        raise VaultSecretError(
            f"Vault path {VAULT_SECRET_PATH} is missing a usable {VAULT_SECRET_FIELD} field"
        )
    return SecretLookupResult(
        api_key=raw_value.strip(),
        source="vault",
        source_detail=f"{VAULT_MOUNT_POINT}/{VAULT_SECRET_PATH}:{VAULT_SECRET_FIELD}",
    )


def load_from_env(env: Mapping[str, str] | None = None) -> SecretLookupResult:
    resolved_env = os.environ if env is None else env
    for variable_name in (ANTHROPIC_API_KEY_ENV, GENERIC_LLM_API_KEY_ENV):
        value = resolved_env.get(variable_name, "").strip()
        if value and not is_placeholder_secret(value):
            return SecretLookupResult(
                api_key=value,
                source="env",
                source_detail=variable_name,
            )
    raise LLMSecretError(
        f"{ANTHROPIC_API_KEY_ENV} is not set and {GENERIC_LLM_API_KEY_ENV} is not set"
    )


def load_llm_api_key(
    *,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory = hvac.Client,
    timeout_seconds: float = 10.0,
) -> SecretLookupResult:
    resolved_env = os.environ if env is None else env
    vault_error: Exception | None = None
    try:
        return load_from_vault(
            env=resolved_env,
            client_factory=client_factory,
            timeout_seconds=timeout_seconds,
        )
    except VaultSecretError as exc:
        vault_error = exc

    try:
        return load_from_env(resolved_env)
    except LLMSecretError as env_error:
        raise LLMSecretError(
            "No usable Anthropic API key found in Vault or environment.\n"
            f"Vault lookup: {vault_error}\n"
            f"Environment lookup: {env_error}\n\n"
            f"{SETUP_INSTRUCTIONS}"
        ) from env_error
