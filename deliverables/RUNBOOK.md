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

## 4.1 RAG corpus + eval (Day 3)

Build the corpus and chunk artifacts (regenerable; gitignored under `data/`):

```bash
# Docs corpus: fetch pinned Airflow .rst, normalize, then issue corpus from the holdout.
uv run --project backend python scripts/build_rag_corpus.py        # raw .rst + manifest
uv run --project backend python scripts/rst_normalize.py           # -> rag_doc_corpus.jsonl
uv run --project backend python scripts/build_issue_corpus.py      # -> rag_issue_corpus.jsonl
# Chunk artifacts (advanced parent-child + naive baseline + manifest).
uv run --directory ml python -m rag.build_chunks
# Golden-set draft (grounded; human-review + 5/25 hand-labels required before freezing).
uv run --directory ml python -m rag.golden
```

Run the RAG evaluation harness:

```bash
uv run --project ml python -m evals.rag_eval --mock           # deterministic offline smoke
uv run --project ml python -m evals.rag_eval --no-generation  # real retrieval metrics (Hit@5/MRR@10)
uv run --project ml python -m evals.rag_eval                  # full ladder + judge (needs ANTHROPIC_API_KEY)
```

Outputs: `artifacts/evals/rag_eval_report.json` and retrieved-chunk snapshots under
`artifacts/evals/rag_retrieved_snapshots/`. CI uses the same harness in deterministic mock mode
with small committed fixtures, writing `artifacts/evals/rag_eval_ci_report.json` so the gate does
not depend on ignored local corpus artifacts, model downloads, Anthropic calls, or secrets.

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
