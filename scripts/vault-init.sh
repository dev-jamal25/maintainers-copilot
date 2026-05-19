#!/bin/sh
set -eu

echo "vault-init: waiting for Vault at ${VAULT_ADDR} ..."
i=0
until vault status >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 30 ]; then
    echo "vault-init: Vault did not become ready within 30s" >&2
    exit 1
  fi
  sleep 1
done

echo "vault-init: seeding fake local-dev secrets under secret/maintainers-copilot/*"

vault kv put secret/maintainers-copilot/database \
  database_url="postgresql+asyncpg://maintainers_copilot@db:5432/maintainers_copilot"

vault kv put secret/maintainers-copilot/minio \
  access_key="example-minio-access-key" \
  secret_key="example-minio-secret-key"

vault kv put secret/maintainers-copilot/jwt \
  secret="example-jwt-signing-key" \
  algorithm="HS256" \
  exp_minutes="60"

vault kv put secret/maintainers-copilot/anthropic \
  api_key="example-anthropic-api-key"

# Langfuse field names MUST stay public_key / secret_key — that's what
# build_tracing_client() reads at backend/app/infra/tracing.py:84-85.
vault kv put secret/maintainers-copilot/langfuse \
  public_key="example-langfuse-public-key" \
  secret_key="example-langfuse-secret-key"

vault kv put secret/maintainers-copilot/wandb \
  api_key="example-wandb-api-key"

echo "vault-init: done"
