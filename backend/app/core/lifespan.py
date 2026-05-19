"""Fail-fast startup checks.

Importable for use from future FastAPI lifespan hooks, and runnable as a
module (python -m app.core.lifespan) so the compose api service can
exercise the check before sleeping.
"""

import sys

from app.core.config import Settings
from app.infra.errors import TracingError, VaultError
from app.infra.tracing import build_tracing_client
from app.infra.vault import VaultClient


def verify_startup_dependencies(settings: Settings | None = None) -> None:
    """Run all boot-time dependency checks in fixed order. Raises on first failure."""
    resolved = settings if settings is not None else Settings()

    vault_client = VaultClient(resolved.vault)
    vault_client.check_ready()

    tracing_client = build_tracing_client(resolved, vault_client)
    tracing_client.check_ready()


def main() -> int:
    try:
        verify_startup_dependencies()
    except VaultError as e:
        print(f"Vault startup check failed: {e}", file=sys.stderr)
        return 1
    except TracingError as e:
        print(f"Tracing startup check failed: {e}", file=sys.stderr)
        return 1
    print("Vault startup check passed", flush=True)
    print("Tracing startup check passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
