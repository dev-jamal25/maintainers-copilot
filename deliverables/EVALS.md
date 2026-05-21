# EVALS.md — Maintainer's Copilot Evaluation Plan

This file starts as a Day 1 evaluation skeleton. Exact metrics and numbers should be filled as the classifier and RAG pipelines are implemented.

# 1. The Golden Sets

## Classification Golden Set

Planned size: 25 hand-curated issues.

Requirements:

- Separate from the model test split.
- Contains issue text and expected label.
- Labels limited to `bug`, `feature`, `docs`, `question`.
- Stable ordering for CI diffing.
- Stored in a reproducible file format, likely JSONL or CSV.

Planned fields:

```text
issue_id
issue_url
title
body
created_at
closed_at
source_labels
expected_class
notes
```

## RAG Golden Set

Size: 25 triples, distributed 8 `issue_only` / 5 `docs_only` / 12 `mixed` (D3.4).

Workflow (draft-first, freeze-after-ingestion):

- `uv run --directory ml python -m rag.golden` builds `data/evals/rag_golden.draft.jsonl` by
  extracting grounded quotes from the chunk artifact (real maintainer comments + doc sections),
  resolving each to stable parent + child chunk IDs. Drafts are `human_reviewed: false`.
- A maintainer refines the grounded ideal answers and hand-labels 5/25 (judge agreement, D1.12),
  then freezes `data/evals/rag_golden.jsonl`.
- The validator (`rag.golden.validate_golden_examples`) enforces the 8/5/12 distribution,
  referential integrity against `rag_chunks.jsonl`, and (for the frozen set) the
  `human_reviewed` gate. Leakage is avoided: golden sources come only from the held-out issues
  and the docs corpus, never from classifier training data.

Fields (final): `question`, `ideal_answer`, `ground_truth_sources`, `ground_truth_parent_ids`,
`ground_truth_child_chunk_ids`, `tags`, `difficulty`, `source_type`, `human_reviewed`.

# 2. Classification Pipeline Evaluation

The classification eval must run against all three models:

1. Classical ML baseline.
2. Fine-tuned DistilBERT-base classifier.
3. LLM baseline.

Metrics to report:

- Accuracy.
- Macro-F1.
- Per-class precision/recall/F1.
- Confusion matrix.
- Latency.
- Approximate cost where applicable.

Planned output:

```text
eval_report.json
classification_confusion_matrix.*
classification_metrics.json
```

Deployment choice must be defended using these numbers, not preference.

# 3. RAG Pipeline Evaluation

Baseline to beat:

```text
naive fixed-size chunking + pure dense retrieval
```

Planned evaluated variants:

1. Naive fixed-size + dense retrieval.
2. Chosen chunking + dense retrieval.
3. Hybrid sparse+dense retrieval.
4. Hybrid + cross-encoder reranking.
5. Hybrid + reranking + multi-query transformation.

Retrieval metrics:

- Hit@5.
- MRR@10.

Generation metrics (frozen Sonnet judge, strict JSON, fail-closed): faithfulness, answer
relevancy, context usefulness, completeness, refusal quality, overall, pass rate.

Also evaluated on the same golden set:

- Embedding comparison: `bge-small-en-v1.5` vs `all-MiniLM-L6-v2` (D3.3).
- Hybrid weight sweep: dense/sparse 0.25/0.75, 0.50/0.50, 0.75/0.25 (D3.8).

Harness: `evals/rag_eval.py` — `uv run --project ml python -m evals.rag_eval`. Model/LLM deps are
injected, so `--mock` runs the whole ladder deterministically offline and `--no-generation` gives
retrieval-only metrics without an API key. Output:

```text
artifacts/evals/rag_eval_report.json        # variants + embedding_comparison + weight_sweep
artifacts/evals/rag_retrieved_snapshots/    # retrieved-chunk snapshots (D3.11)
```

# 4. LLM-as-a-Judge Alignment

Chosen approach: frozen judge model.

Generator/chatbot model:

```text
Claude Haiku 4.5
```

Frozen judge model:

```text
Claude Sonnet 4.6
```

Planned judge rubric dimensions:

- Faithfulness to retrieved context.
- Answer relevancy to the user question.
- Refusal when context is insufficient.
- No unsupported claims.

Human alignment check:

- Hand-label 5 of the 25 RAG examples.
- Compare human labels against judge outputs.
- Report agreement.
- If disagreement appears, document whether the judge, rubric, or human label changed.

# 5. Continuous Integration (CI) Thresholds

Thresholds will live in:

```text
eval_thresholds.yaml
```

Planned CI gates:

- Lint.
- Type-check.
- Build images.
- Classification eval.
- RAG eval.
- Redaction test.
- Stack smoke test.

Rules:

- Eval thresholds cannot be zero.
- Eval thresholds cannot be disabled.
- `eval_report.json` is written every run.
- `eval_report.json` is stored in MinIO.
- Regression below threshold blocks merge.

Initial thresholds should only be frozen after the first real baseline run.

RAG status (Day 3): the real local RAG eval has been completed on the frozen golden set, and
the selected candidate is `parent_child_hybrid` with `bge-small-en-v1.5` at dense 0.50 /
sparse 0.50. CI runs a dedicated **`rag-eval`** job in deterministic `--mock` mode against
small committed fixtures, so it validates the harness path without Anthropic calls, model
downloads, secrets, or gitignored local corpus artifacts. The full local real eval remains the
source of the reported RAG quality numbers.
