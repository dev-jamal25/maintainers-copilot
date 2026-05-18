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

Planned size: 25 question / ideal-answer / ground-truth-chunks triples.

Requirements:

- Questions based on project docs and held-out resolved issues.
- Ground-truth chunks must be known.
- Held-out RAG issues must not leak into classifier training.
- Stable ordering for CI diffing.

Planned fields:

```text
question_id
question
ideal_answer
ground_truth_chunk_ids
required_metadata_filters
notes
```

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

Generation metrics:

- Faithfulness.
- Answer relevancy.

Planned output:

```text
eval_report.json
rag_retrieval_metrics.json
rag_generation_metrics.json
retrieved_chunks_snapshot.json
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
