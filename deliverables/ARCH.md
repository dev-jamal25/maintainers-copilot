# ARCH.md — Maintainer's Copilot Architecture

## 1. Project Overview

Maintainer's Copilot is an authenticated issue-triage assistant for open-source maintainers. The system helps a maintainer classify issues, extract useful code-shaped entities, summarize issue threads, retrieve grounded answers from project documentation and resolved issues, and preserve useful memory across conversations.

The project has three main technical tracks:

1. **Deep Learning / NLP track**
   - Classical ML baseline for issue classification.
   - Fine-tuned DistilBERT-base classifier.
   - LLM baseline classifier.
   - NER endpoint for code-shaped entities.
   - Summarization endpoint for issue threads.

2. **Advanced RAG track**
   - Corpus from project documentation and held-out resolved issues.
   - Non-naive chunking strategy.
   - Dense + sparse hybrid retrieval.
   - Cross-encoder reranking.
   - Multi-query transformation.
   - Metadata filtering.
   - Frozen-judge evaluation.

3. **Chatbot + memory + embeddable widget track**
   - Single tool-calling LLM.
   - Streamlit authenticated admin/user dashboard.
   - React/Vite embeddable widget.
   - Short-term memory in Redis.
   - Long-term episodic memory in Postgres + pgvector.
   - Explicit `write_memory` tool.

## 2. Frozen Day 1 Technology Choices

| Area | Decision |
|---|---|
| Dataset source | HuggingFace Transformers GitHub issues |
| Long-term memory | Episodic memory |
| Tracing backend | Langfuse |
| Fine-tuning model | DistilBERT-base |
| Fine-tuning run logger | Weights & Biases |
| Embedding model | `bge-small-en-v1.5` |
| Reranker | `ms-marco-MiniLM` cross-encoder |
| Query transformation | Multi-query transformation |
| Chatbot LLM | Anthropic Claude Haiku 4.5 |
| Frozen judge | Claude Sonnet 4.6 |
| Streaming | Server-Sent Events with named events |
| Auth scope | Stick to brief: admin-invite-only, no password reset/email verification for MVP |

## 3. Service Architecture

The docker-compose stack will contain these services:

| Service | Responsibility |
|---|---|
| `api` | FastAPI backend for auth, chat orchestration, memory, RAG, widget config, and admin APIs |
| `chatbot` | Streamlit dashboard for authenticated users/admins |
| `widget` | Static server for the built React widget bundle and loader script |
| `model-server` | FastAPI inference server for classifier, NER, and summarizer |
| `host` | Nginx demo host app that embeds the widget |
| `migrate` | Alembic migration container that exits before API boot |
| `db` | Postgres 16 + pgvector |
| `redis` | Short-term memory and cache |
| `minio` | Blob storage for eval reports, model artifacts/manifests, plots, and retrieved chunk snapshots |
| `vault` | HashiCorp Vault dev service for secrets |

## 4. Backend Layering Rules

The backend must stay clean because the architecture is part of the grade.

```text
app/
  api/              # HTTP routes only: request/response handling
  services/         # business logic, transactions, cache/memory invalidation
  repositories/     # SQL queries only
  domain/           # Pydantic domain models and domain exceptions
  infra/            # adapters: Vault, Redis, MinIO, LLM, model-server, tracing, redaction
  db/               # SQLAlchemy ORM models, session, Alembic migrations
  core/             # settings, logging, app-level boot checks
  prompts/          # version-controlled prompt files
```

Rules:

- `app/api/` must not touch SQLAlchemy, Redis, MinIO, Vault, or external clients directly.
- `app/services/` owns business workflows and transaction boundaries.
- `app/repositories/` owns SQL only and does not raise HTTP errors.
- `app/domain/` stays independent from ORM models.
- `app/infra/` wraps external systems and runs redaction before logs, traces, or memory writes.

## 5. Main Runtime Flow

### 5.1 Authenticated Streamlit Chat Flow

```text
User logs in through Streamlit
  -> Streamlit sends JWT-authenticated request to FastAPI
  -> API validates user and role
  -> Chat service loads short-term state from Redis
  -> Single tool-calling LLM decides whether to call tools
  -> Tools call model-server / RAG / memory service over typed contracts
  -> API streams response using SSE named events
  -> Trace spans are recorded in Langfuse
  -> Logs include request_id + trace_id after redaction
```

### 5.2 Tool Flow

The chatbot is a single tool-calling LLM. It is not a multi-agent workflow.

Available tools planned for MVP:

| Tool | Backend capability |
|---|---|
| `classify_issue` | Calls model-server classifier endpoint |
| `extract_entities` | Calls model-server NER endpoint |
| `summarize_thread` | Calls model-server summarizer endpoint |
| `rag_search` | Calls RAG service to retrieve and answer from docs/issues |
| `write_memory` | Explicitly writes long-term episodic memory and audit log row |

Tool failures must return structured tool errors to the LLM. A failed classifier endpoint should not crash the conversation.

## 6. Model Server

`model-server` is separate from `api` to keep inference concerns isolated.

Planned endpoints:

```text
POST /classify
POST /ner
POST /summarize
GET  /health
```

The API and chatbot call these endpoints over HTTP using async clients with timeouts and bounded retries.

## 7. Data and Splits

Dataset source: HuggingFace Transformers GitHub issues.

Planned label mapping:

| Project labels | Target class |
|---|---|
| bug-related labels | `bug` |
| enhancement / feature request labels | `feature` |
| documentation labels | `docs` |
| question / usage / help labels | `question` |

The exact GitHub labels will be frozen after the first scrape and written in `DECISIONS.md`.

Split policy:

- Train/validation/test are based on issue creation or closing time.
- Test must be strictly more recent than train.
- RAG held-out resolved issues must not appear in classifier training.
- Golden sets are hand-curated and separate from training data.

## 8. RAG Architecture

Baseline to beat:

```text
fixed-size chunks + pure dense retrieval
```

Planned production RAG path:

```text
User question
  -> multi-query transformation
  -> sparse retrieval + dense retrieval
  -> weighted hybrid merge
  -> metadata filtering
  -> cross-encoder reranking
  -> answer generation with grounded prompt
  -> retrieved chunk snapshot stored in MinIO for last N conversations
```

Planned vector store: Postgres + pgvector.

Planned embedding model: `bge-small-en-v1.5`.

Planned reranker: `ms-marco-MiniLM` cross-encoder.

## 9. Memory Architecture

### Short-Term Memory

- Store conversation state in Redis.
- TTL must be explicit and justified after implementation.
- Used for active conversation continuity.

### Long-Term Memory

Chosen type: episodic memory.

Reason:

- Easiest to demonstrate cross-conversation recall by storing events like: “user asked about issue classification trade-offs on Monday”.
- Fits maintainer workflow because past triage decisions and prior conversations are chronological events.
- Less scope-heavy than semantic or procedural memory for a five-day solo project.

Every long-term write must create an audit-log row with actor, action, target, and timestamp.

## 10. Widget Architecture

The widget is a production-shaped embeddable React app, separate from the Streamlit dashboard.

Planned flow:

```text
Host page includes:
<script src="/widget.js" data-widget-id="..."></script>

/widget.js
  -> reads data-widget-id
  -> fetches widget config from API
  -> validates allowed origin
  -> injects iframe pointing to React widget bundle
  -> sets postMessage channel for iframe resize
```

Widget configuration lives in Postgres:

| Field | Purpose |
|---|---|
| `widget_id` | Public identifier |
| `allowed_origins` | Per-widget origin allowlist |
| `theme` | Runtime styling configuration |
| `greeting` | Initial greeting |
| `enabled_tools` | Which tools the widget can expose |

Security requirements:

- CORS allowlist comes from DB widget config, not hardcoded env.
- Embed route sets `Content-Security-Policy: frame-ancestors ...` based on allowed origins.
- Friday demo should show allowed host working and unallowed host blocked by browser.

## 11. Observability

Tracing backend: Langfuse.

Required traced spans:

- LLM calls.
- Tool calls.
- RAG retrieval.
- Reranking.
- Memory writes.
- Error paths.

Every trace should be joinable with structured logs using `trace_id` and `request_id`.

## 12. Refuse-to-Boot Policy

The API should refuse to boot if:

- Vault is unreachable.
- Required secrets cannot be resolved.
- Classifier weights are missing.
- Classifier weights SHA-256 does not match the model card.
- Tracing backend is misconfigured.
- Any committed eval threshold is zero or disabled.

## 13. Day-by-Day Implementation Plan

### Monday — Foundations

- Repo skeleton.
- Docker-compose services.
- Vault wiring.
- Tracing wiring from day one.
- Alembic baseline.
- Dataset fetch and split script.
- Start DistilBERT fine-tuning.
- Create project deliverable Markdown files.

### Tuesday — DL Track

- Finish classifier.
- Model card.
- Classical ML baseline.
- LLM baseline.
- Three-way comparison.
- Classification golden set.
- NER and summarization endpoints.

### Wednesday — Advanced RAG

- Corpus ingestion.
- Chunking strategy.
- Embedding comparison.
- Hybrid retrieval.
- Reranking.
- Multi-query transformation.
- RAG golden set.
- Redaction layer and exception handling.

### Thursday — Chatbot, Memory, Embed

- Auth.
- Streamlit chatbot and admin dashboard.
- Tool-calling loop.
- Short-term and long-term memory.
- Widget config.
- React widget bundle.
- Loader script.
- Host app.
- CI eval gates.

### Friday — Polish and Presentation

- Final integration.
- CI green.
- README and deliverables finalized.
- Demo practice.
- Tag release `v0.1.0-week7`.
