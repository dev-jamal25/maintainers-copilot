# GitHub Copilot Instructions — Maintainer's Copilot (Week 7)

This is the Week 7 AIE Bootcamp Maintainer's Copilot repository. Copilot suggestions must respect the architecture in `CLAUDE.md`. Summary of the rules below — when in doubt, read `CLAUDE.md`.

## Backend layer boundaries (strict)

- `backend/app/api/` — HTTP routes only. No SQLAlchemy, Redis, MinIO, Vault, LLM, or external client calls.
- `backend/app/services/` — business logic, transactions, cache/memory invalidation, orchestration.
- `backend/app/repositories/` — SQL only. No HTTP errors. No external calls.
- `backend/app/domain/` — Pydantic domain models, enums, domain exceptions.
- `backend/app/infra/` — adapters for Vault, Redis, MinIO, LLM, tracing, redaction, model-server. Redaction runs here before data reaches logs/traces/memory.
- `backend/app/db/` — SQLAlchemy ORM models, sessions, Alembic migrations.
- `backend/app/core/` — settings, logging, app lifespan, boot checks.
- `backend/prompts/` — version-controlled prompt files (no large prompts inside Python source).

## Language and tooling

- Python 3.12. Manage deps with `uv`, never pip.
- All I/O is async. Use `httpx.AsyncClient`, never `requests`, in server paths.
- Use `asyncio.sleep()`, never `time.sleep()`, in async paths.
- Type hints on all functions. Pydantic models at external boundaries.
- Use FastAPI dependency injection — do not instantiate clients inside routes.
- No `print()` in app code; use structured logging.

## Security

- Never hardcode secrets. All runtime secrets resolve from Vault at startup.
- Never commit `.env`, API keys, model weights, datasets, or large artifacts.
- JWT payloads contain no sensitive data.
- Redaction runs before logs, traces, and memory writes.

## Git workflow

- Branches use `feat/`, `fix/`, `chore/`, `docs/` — never `feature/`.
- Conventional commits (e.g., `feat(api): add chat SSE endpoint`).
- Keep PRs small and scoped.

## Out of scope for any single suggestion

Do not generate implementation code outside the active task scope. If the user is editing `backend/app/api/`, do not silently add SQL queries, ORM models, or external client code — those belong in `services/`, `repositories/`, and `infra/` respectively.
