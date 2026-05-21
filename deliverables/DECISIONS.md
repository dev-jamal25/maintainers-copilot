# DECISIONS.md — Maintainer's Copilot

This file records project decisions as they are frozen during the week. Each day has its own section so the reasoning stays chronological and easy to defend during review.

# Day 1 decisions

## D1.1 Dataset source: apache/airflow GitHub issues

**Decision:** Use closed issues from the Apache Airflow repository.

**Reasoning:**

- The repository has a large and active issue history with enough closed issues for a balanced classifier dataset.
- Its labels map cleanly into the required four classes: `kind:bug`, `kind:feature`, `kind:documentation`, and `pending-response` as the documented question heuristic.
- Issues are operationally rich and technical enough for meaningful entity extraction, summarization, and later RAG evaluation.
- Fetching by class label gives a reproducible 500-example target per class without faking or duplicating examples.

**Implementation note:**

We will scrape a bounded number of class-labeled issues locally, export the mapped dataset, and train/split on Colab to avoid local compute bottlenecks.

**Still to freeze after scraping:**

- Final Colab train/validation/test counts.

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

## D1.16 Carve-out reserve selection

**Decision:** Add a standalone reserve-carving script before classifier train/validation/test splitting.

**Selection criteria:**

- Classification golden eval: choose 25 short, unambiguous mapped issues, balanced as 7 `bug`, 6 `docs`, 7 `feature`, and 5 `question`, where the class is clear from the title and opening paragraph.
- RAG holdout: choose 50-100 mapped issues only when the raw issue record includes thread comments with a substantive maintainer-side response from a `MEMBER`, `OWNER`, or `CONTRIBUTOR`; the response must be at least 80 words, with preference for code blocks, lists, or numbered steps.

**Rationale:**

The classifier golden set and RAG holdout must be excluded from classifier training so later evaluation is not contaminated by examples the model has already seen. The golden eval prioritizes obvious labels for stable classifier sanity checks. The RAG holdout requires maintainer comments because the retrieval/generation task should evaluate answers grounded in real maintainer responses, not issue descriptions alone.

**Final counts from the comment-enriched run:**

- Classification golden eval: 25 issues, with 7 `bug`, 6 `docs`, 7 `feature`, and 5 `question`.
- RAG holdout: 100 issues, with 22 `bug`, 20 `docs`, 14 `feature`, and 44 `question`.
- Splittable classifier pool: 1855 issues, with 471 `bug`, 454 `docs`, 479 `feature`, and 451 `question`.

The RAG holdout was selected only after re-fetching the raw issue pool with GitHub issue comments into `data/raw/github_issues_with_comments.jsonl`. These reserve files are excluded from classifier training.

## D1.17 Data-quality cleaning pass

**Decision:** Apply a deterministic quality filter to the mapped classifier dataset before reserve
carving.

**Filter rules and rationale:**

- Drop `char_count < 20` because shorter records do not carry enough context for a
  useful issue classifier example.
- Drop `word_count < 5` because fewer than five lexical tokens usually means a
  placeholder, fragment, or accidental issue body.
- Drop only-punctuation text because it has no alphanumeric signal for label learning or evaluation.
- Drop obvious junk patterns (`.`, `.\n\n.`, `..`, `n/a`, `na`, `test`, `delete`) because
  they are explicit placeholders rather than maintainer-triage examples.

**Counts:**

- Rows in: 1980.
- Rows out: 1973.
- Dropped per class: {"bug": 1, "docs": 1, "feature": 1, "question": 4}.

`data/processed/issues_mapped.dirty.jsonl` preserves the pre-cleaning mapped dataset for
auditability.

# Day 2 decisions

## D2.2 Classification deployment candidate: classical baseline

**Decision:** Use the classical TF-IDF + LogisticRegression baseline as the current deployment candidate for issue classification.

**Reasoning:**

- It has the strongest full-test macro-F1 among the completed local baselines.
- It is far faster than the selected DistilBERT candidate and the LLM baseline.
- It has a small local artifact and no per-request inference cost.
- The selected DistilBERT candidate remains useful for transformer comparison, but it does not beat the classical baseline yet.
- The LLM baseline remains useful as an audit/comparison baseline, but the Phase 6B balanced sample is slower and weaker, especially on `question`.

**Recommendation summary:** Choose the classical baseline for now because it has the highest test macro-F1, the lowest latency, and a small local artifact. DistilBERT remains useful as a transformer baseline, and the LLM baseline is useful for audit comparison but is slower and weaker on the balanced sample.


# Day 3 decisions
## D3.1 RAG docs corpus scope

**Decision:** Use a bounded Airflow documentation corpus selected from recurring themes in `rag_holdout.jsonl`, not the full Airflow docs site.

**Scope selection method:** Inspect the 100 held-out resolved issues, identify recurring maintainer-support themes, and ingest only docs that support those themes.

**Included areas:** installation/environment, Core Concepts, DAGs, tasks, scheduling, scheduler/executor behaviour, operators/hooks, configuration, connections, variables, CLI, REST/public API, database migrations, troubleshooting, FAQ, best practices, and relevant how-to guides.

**Provider docs policy:** Exclude broad provider docs by default because they are too large and would dilute retrieval. Add provider docs only when the RAG golden set proves they are needed. Kubernetes-related docs are the only likely early exception.

**Reasoning:** The corpus should match realistic Airflow maintainer questions from the held-out issues, not every possible Airflow topic.

## D3.2 Chunking strategy

**Decision:** Use hierarchical parent-child chunking.

**Parent chunks:** Airflow markdown/header sections, documentation source sections, or issue/comment-level source blocks.

**Child chunks:** smaller retrievable chunks generated from each parent.

**Default parameters:** parent = natural section; child = 350–500 tokens; overlap = 50–75 tokens.

**Reasoning:** Parent-child chunking gives precise retrieval while preserving enough parent context for grounded answer generation. It is the non-naive chunking strategy used to beat the fixed-size baseline.

## D3.3 Embedding comparison

**Decision:** Use `bge-small-en-v1.5` as the planned production embedding model and compare it against `all-MiniLM-L6-v2`.

**Reasoning:** `bge-small-en-v1.5` is the intended stronger local embedding model. `all-MiniLM-L6-v2` is free, small, fast, and easy to run as a realistic baseline.

**Validation rule:** The final embedding choice must be backed by Hit@5 and MRR@10 on the RAG golden set.

## D3.4 RAG golden triples

**Decision:** Build 25 RAG golden triples using a draft-first, finalize-after-ingestion workflow.

**Draft stage:** Create draft examples from `rag_holdout.jsonl` and matched Airflow docs before chunking.

**Draft fields:** `question`, `ideal_answer`, `ground_truth_sources`, `tags`, `difficulty`, and `source_type`.

**Source types:** `issue_only`, `docs_only`, and `mixed`.

**Distribution target:** 8 issue-only, 5 docs-only, and 12 mixed issue + docs examples.

**Grounding rule:** Ideal answers must come from maintainer comments, Airflow docs, or both. Do not write generic Airflow advice from memory.

**After ingestion:** Resolve source references into stable parent and child chunk IDs.

**Final file:** `data/evals/rag_golden.jsonl`.

**Final fields:** `question`, `ideal_answer`, `ground_truth_sources`, `ground_truth_parent_ids`, `ground_truth_child_chunk_ids`, `tags`, `difficulty`, and `source_type`.

**Freeze policy:** Freeze the golden set before retrieval tuning. Only edit later for factual errors, duplicate examples, invalid examples, or broken source/chunk IDs.

## D3.5 Judge rubric prompt

**Decision:** Use a versioned frozen judge rubric prompt stored at `backend/prompts/rag_judge_rubric_v1.md`.

**Judge model:** Claude Sonnet 4.6.

**Generator model:** Claude Haiku 4.5.

**Rubric rule:** The judge evaluates only the provided question, retrieved context, generated answer, ideal answer, and chunk IDs. It must not use outside Airflow knowledge.

**Scored dimensions:** faithfulness, answer relevancy, context usefulness, completeness, refusal quality, and overall score.

**Output:** strict JSON only.

**Pass rule:** Normal answers pass only when faithfulness, answer relevancy, and overall are each at least 4/5. Insufficient-context answers pass only when refusal quality is at least 4/5 and there are no unsupported claims.

## D3.6 Metadata schema

**Decision:** Attach a minimal stable metadata schema to every chunk before indexing.

**Fields:** `chunk_id`, `parent_id`, `source_type`, `source_id`, `title`, `url`, `section_path`, `airflow_area`, `tags`, `version`, `github_issue_id`, `github_comment_id`, `created_at`, and `closed_at`.

**Reasoning:** Metadata is required for filtering, debugging, golden-set references, and explaining retrieval results.

## D3.7 Chunk ID policy

**Decision:** Use stable, human-readable IDs for parent and child chunks.

**Patterns:**
- `doc:{source_id}:parent:{n}`
- `doc:{source_id}:parent:{n}:child:{m}`
- `issue:{github_id}:comment:{comment_id}:parent:{n}`
- `issue:{github_id}:comment:{comment_id}:child:{m}`

**Reasoning:** Stable IDs are required for `rag_golden.jsonl`, retrieval metrics, rerun consistency, and debugging.

## D3.8 Hybrid retrieval weighting

**Decision:** Do not guess the final sparse/dense retrieval weight manually. Sweep a small set and pick the best result on the golden set.

**Initial sweep:** dense 0.25 / sparse 0.75, dense 0.50 / sparse 0.50, and dense 0.75 / sparse 0.25.

**Selection metrics:** Hit@5 and MRR@10.

## D3.9 Multi-query transformation

**Decision:** Use 3 query rewrites by default.

**Reasoning:** Three rewrites are enough to improve recall without making latency and LLM cost too high for the MVP.

## D3.10 Retrieval defaults

**Decision:** Use standard defaults first, then tune only if evals show weakness.

**Defaults:** dense top_k = 20, sparse top_k = 20, hybrid merged top_k = 30, rerank top_k = 10, final context chunks = 5.

## D3.11 Retrieved chunk snapshots

**Decision:** Store retrieved-chunk snapshots for the last 50 conversations.

**Storage:** MinIO.

**Format:** JSON.

**Reasoning:** This gives enough traceability for debugging and demo evidence without overbuilding retention.

## D3.12 RAG eval thresholds

**Decision:** Do not freeze RAG thresholds before the first real baseline run.

**Reasoning:** Thresholds must be non-zero and based on measured results. First run the naive baseline and the initial advanced RAG variant, then set meaningful CI thresholds.

**Decision:** Use `bge-small-en-v1.5` with the `parent_child_hybrid` retrieval pipeline as the current Day 3 RAG candidate.

**Embedding comparison:**
- `bge-small-en-v1.5`: Hit@5 = 0.36, MRR@10 = 0.218.
- `all-MiniLM-L6-v2`: Hit@5 = 0.32, MRR@10 = 0.180.

**Hybrid weighting result:**
The selected hybrid weighting is dense 0.50 / sparse 0.50. This achieved the best Hit@5 at 0.36. Dense 0.75 / sparse 0.25 had a nominally higher MRR@10, but the difference was too small to justify sacrificing the stronger Hit@5.

**Chosen variant:**
`parent_child_hybrid`.

**Final selected pipeline metrics:**
- Hit@5 = 0.36.
- MRR@10 = 0.218.
- Faithfulness = 4.52.
- Answer relevancy = 4.28.
- Judge pass rate = 0.44.

**Reasoning:**
The chosen pipeline gives the best balance of retrieval success and answer quality. The 44% judge pass rate shows there is still room to improve recall and context coverage, but the faithfulness and answer relevancy scores indicate that when useful context is retrieved, the generated answers are well grounded.

# Day 4 decisions

_TODO: Add Thursday chatbot/memory/widget/CI decisions after implementation._

# Day 5 decisions

_TODO: Add Friday final deployment, demo, and release decisions._