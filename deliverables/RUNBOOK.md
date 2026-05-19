# RUNBOOK.md — Maintainer's Copilot

This runbook is intentionally a skeleton on Day 1. It should be filled as implementation becomes real. Do not invent commands before they exist in the repository.

# 1. Prerequisites (The "Before You Start" Checklist)

_TODO: Fill after the repo skeleton and dependency managers are finalized._

Expected areas to document later:

- Docker / Docker Compose version.
- Python version.
- uv setup.
- Node version for widget build.
- Vault dev token setup.
- Required `.env` creation from `.env.example`.

# 2. Startup and Shutdown Procedures

_TODO: Fill once `docker-compose.yml` is implemented._

Expected areas to document later:

- Fresh clone startup.
- Migration container behaviour.
- Starting the full stack.
- Stopping the full stack.
- Resetting local volumes, if needed.

# 3. Secrets and Environment Variables

_TODO: Fill after Vault wiring is implemented._

Expected areas to document later:

- Which values are allowed in `.env`.
- Which values must live in Vault.
- How to seed Vault locally.
- How to verify the API resolved required secrets.

# 4. Routine Operations (The "How-To" Guide)

_TODO: Fill as features land._

Expected areas to document later:

- Run dataset scrape.
- Run classifier training.
- Run classification eval.
- Run RAG ingestion.
- Run RAG eval.
- Create an admin invite.
- Create/edit widget config.
- Build widget bundle.
- Tag release.

# 5. Monitoring and Access (Where to Look)

_TODO: Fill after observability services are wired._

Expected areas to document later:

- FastAPI docs URL.
- Streamlit dashboard URL.
- Widget demo host URL.
- Langfuse URL.
- MinIO console URL.
- Vault URL.
- Database access method.
- Logs and trace IDs.

# 6. Troubleshooting / Common Pitfalls

_TODO: Fill with real issues encountered during implementation._

Expected areas to document later:

- Vault unreachable.
- Missing model artifact.
- SHA-256 mismatch.
- Langfuse misconfiguration.
- Eval thresholds disabled.
- Widget blocked by CSP/frame-ancestors.
- Redis connection failure.
- MinIO upload failure.
