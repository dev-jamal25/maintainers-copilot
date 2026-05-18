"""Fail-fast startup checks.

Importable for use from future FastAPI lifespan hooks, and runnable as a
module (python -m app.core.startup_checks) so the compose api service can
exercise the check before sleeping.
"""

import sys

from app.core.settings import VaultSettings
from app.infra.errors import VaultError
from app.infra.vault import VaultClient


def check_vault_ready(settings: VaultSettings | None = None) -> None:
    """Construct a VaultClient and verify Vault is ready. Raises VaultError on failure."""
    resolved = settings if settings is not None else VaultSettings()
    client = VaultClient(resolved)
    client.check_ready()


def main() -> int:
    try:
        check_vault_ready()
    except VaultError as e:
        print(f"Vault startup check failed: {e}", file=sys.stderr)
        return 1
    print("Vault startup check passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
