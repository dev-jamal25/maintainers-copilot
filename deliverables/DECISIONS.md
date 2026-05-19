# DECISIONS.md — Maintainer's Copilot

This file records project decisions as they are frozen during the week. Each day has its own section so the reasoning stays chronological and easy to defend during review.

# Day 1 decisions

## D1.1 Dataset source: HuggingFace Transformers GitHub issues

**Decision:** Use closed issues from the HuggingFace Transformers repository.

**Reasoning:**

- The repository has a large and active issue history.
- Issues are technical enough for meaningful entity extraction.
- Labels are likely rich enough to map into `bug`, `feature`, `docs`, and `question`.
- The domain fits the bootcamp project because it is an ML/NLP open-source project.

**Implementation note:**

We will scrape a bounded number of issues and train on Colab to avoid local compute bottlenecks.

**Still to freeze after scraping:**

- Exact number of scraped issues.
- Exact GitHub label mapping.
- Final train/validation/test counts.

## D1.2 Classification label mapping policy

**Decision:** Map maintainer-applied labels into four classes: `bug`, `feature`, `docs`, and `question`.

**Reasoning:**

The project brief requires these four classifier outputs. We will keep the mapping explicit because label mapping is part of the dataset definition and must be defendable.

**Initial planned mapping:**

| Target class | Expected source labels |
|---|---|
| `bug` | bug, regression, crash, broken behaviour |
| `feature` | enhancement, feature request, proposal |
| `docs` | documentation, docs, typo in docs |
| `question` | question, usage, help wanted / usage support |

**Freeze condition:**

The exact mapping is frozen only after inspecting real repository labels.

## D1.3 Split policy: time-aware and leakage-safe

**Decision:** Use time-aware splits where the test set is strictly more recent than the train set.

**Reasoning:**

A maintainer assistant is evaluated on future issues, not randomly shuffled historical issues. Time-aware splitting gives a more realistic estimate of generalization.

**Leakage rule:**

Held-out RAG issues must not appear in classifier training.

## D1.4 Long-term memory type: episodic

**Decision:** Use episodic long-term memory.

**Reasoning:**

- It is the simplest useful memory type for a five-day solo project.
- It supports a clear Friday demo: cross-conversation recall.
- It maps naturally to maintainer work: “what was discussed, when, and in what context”.
- It avoids the extra extraction/normalization scope required by semantic memory.
- It avoids the complexity of procedural memory, which would require learning workflows and preferences.

**Storage:**

Postgres + pgvector.

**Audit requirement:**

Every long-term memory write creates an audit-log row.

## D1.5 Tracing backend: Langfuse

**Decision:** Use Langfuse for tracing.

**Reasoning:**

- It is product-shaped and easy to explain in a demo.
- It supports LLM traces, tool calls, metadata, and observability workflows.
- It fits the requirement to show real trace trees with error paths.
- It is more directly LLM-oriented than generic tracing alone.

**Implementation rule:**

Every LLM call, tool call, and RAG retrieval must create a span. Span inputs and outputs must be redacted before they leave the service boundary.

## D1.6 Fine-tuning base model: DistilBERT-base

**Decision:** Fine-tune DistilBERT-base for issue classification.

**Reasoning:**

- Small enough for Colab and fast iteration.
- Strong enough to beat a basic classical baseline in many text classification tasks.
- Easier to train and defend than larger transformer models under a tight deadline.
- Good deployment story for latency and model size.

**Model card must include:**

- Architecture.
- Hyperparameters.
- Training data hash.
- Final metrics.
- Freeze policy.
- Artifact hash.

## D1.7 Fine-tuning run logger: Weights & Biases

**Decision:** Use Weights & Biases for fine-tuning run logging.

**Reasoning:**

- Fast setup for Colab.
- Good visual tracking for loss, metrics, and training runs.
- Cleaner fine-tune story than only local JSON files.

**Artifact policy:**

Final exported model files and model card should still be stored in project-compatible artifact locations, with MinIO used by the system for model/eval artifacts where applicable.

## D1.8 Embedding model: bge-small-en-v1.5

**Decision:** Use `bge-small-en-v1.5` as the planned RAG embedding model.

**Reasoning:**

- Open-source and reproducible.
- Small enough for a local/model-server setup.
- Avoids relying on an external embedding API for core retrieval.
- Good fit for short technical issue/document chunks.

**Validation requirement:**

This remains a planned choice until it is backed by a retrieval-quality number against at least one alternative on the RAG golden set.

## D1.9 Reranker: ms-marco-MiniLM cross-encoder

**Decision:** Use an `ms-marco-MiniLM` cross-encoder as the reranker.

**Reasoning:**

- Fast and classic cross-encoder option.
- Lower compute burden than larger rerankers.
- Easy to explain: first retrieve broadly, then rerank top candidates with a stronger pairwise relevance model.

**Validation requirement:**

Reranking must be evaluated against the naive baseline and hybrid retrieval before being defended as a final choice.

## D1.10 Query transformation: multi-query transformation

**Decision:** Use multi-query transformation.

**Reasoning:**

- Maintainer questions can be vague or use different wording from the docs/issues.
- Multiple rewrites improve recall by searching the same intent from several angles.
- Easier to implement and explain than a heavier RAG-Fusion setup.

**Cost control:**

Limit the number of generated rewrites. Log transformation latency and token cost.

## D1.11 Chatbot LLM: Claude Haiku 4.5

**Decision:** Use Anthropic Claude Haiku 4.5 as the chatbot generator/tool-calling LLM.

**Reasoning:**

- Lower cost and latency than a larger model.
- Good fit for an assistant that mostly chooses tools and synthesizes tool outputs.
- Keeps the heavier model reserved for evaluation.

**Prompt policy:**

Prompts must live in `prompts/` as version-controlled files, not inline strings scattered through the codebase.

## D1.12 RAG evaluator: frozen judge model

**Decision:** Use a frozen judge model for RAG generation evaluation.

**Models:**

- Generator/chatbot: Claude Haiku 4.5.
- Evaluator/frozen judge: Claude Sonnet 4.6.

**Reasoning:**

- Gives more control over the rubric than a framework-only metric.
- Easier to explain exactly what the judge is checking.
- Keeps the evaluation model separate from the generation model.

**Honesty check:**

Hand-label 5 of the 25 RAG golden examples and report agreement with the judge.

## D1.13 Streaming: SSE with named events

**Decision:** Use Server-Sent Events with named events for chat streaming.

**Reasoning:**

- The chat stream is server-to-client, so SSE is simpler than WebSockets.
- Named events can separate token output, tool status, retrieval status, errors, and completion.
- Easier to integrate with both Streamlit/API testing and a React widget.

**Planned event names:**

```text
event: token
event: tool_started
event: tool_finished
event: retrieval_snapshot
event: memory_written
event: error
event: done
```

## D1.14 Auth scope: stick to the brief

**Decision:** Implement email/password + JWT with admin-invite-only user creation. Do not add password reset or email verification for MVP.

**Reasoning:**

- The project is already large for five days.
- The brief asks for admin invites and two roles.
- Password reset/email verification add mail infrastructure and edge cases that are not core to the grading requirements.

## D1.15 Documentation strategy

**Decision:** Start the required deliverables on Day 1:

- `ARCH.md`
- `DECISIONS.md`
- `RUNBOOK.md`
- `EVALS.md`
- `SECURITY.md`

**Reasoning:**

Writing decisions early prevents vague Friday explanations. The files can start as skeletons and become more concrete as code and metrics land.

# Day 2 decisions

_TODO: Add Tuesday DL/classifier/model-card decisions after implementation._

# Day 3 decisions

_TODO: Add Wednesday RAG/redaction/exception decisions after implementation._

# Day 4 decisions

_TODO: Add Thursday chatbot/memory/widget/CI decisions after implementation._

# Day 5 decisions

_TODO: Add Friday final deployment, demo, and release decisions._
