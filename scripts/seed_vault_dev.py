"""Local-dev only. Seeds fake placeholder values into Vault KV v2.

Not used by tests. Not safe for any non-dev environment. The values are
obvious placeholders; rotating these does not affect any real system.

Usage (after compose up):
    VAULT_ADDR=http://localhost:8200 \\
    VAULT_DEV_ROOT_TOKEN_ID=dev-only-root-token \\
    uv run python scripts/seed_vault_dev.py
"""

import os
import sys

import hvac

FAKE_SECRETS: dict[str, dict[str, str]] = {
    "maintainers-copilot/database": {"password": "example-db-password"},
    "maintainers-copilot/jwt": {"signing_key": "example-jwt-signing-key"},
    "maintainers-copilot/anthropic": {"api_key": "example-anthropic-key"},
    "maintainers-copilot/langfuse": {
        "secret_key": "example-langfuse-secret",
        "public_key": "example-langfuse-public",
    },
    "maintainers-copilot/wandb": {"api_key": "example-wandb-key"},
    "maintainers-copilot/minio": {
        "root_user": "example-minio-user",
        "root_password": "example-minio-password",
    },
}


def main() -> int:
    addr = os.environ.get("VAULT_ADDR", "")
    token = os.environ.get("VAULT_DEV_ROOT_TOKEN_ID", "")
    if not addr or not token:
        print(
            "ERROR: set VAULT_ADDR and VAULT_DEV_ROOT_TOKEN_ID before running.",
            file=sys.stderr,
        )
        return 1

    client = hvac.Client(url=addr, token=token)
    if not client.is_authenticated():
        print(f"ERROR: Vault authentication failed at {addr}", file=sys.stderr)
        return 1

    for path, data in FAKE_SECRETS.items():
        client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data,
            mount_point="secret",
        )
        print(f"seeded: secret/{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
