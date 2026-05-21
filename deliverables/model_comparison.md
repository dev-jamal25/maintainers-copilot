# Classification Model Comparison

Phase 7 compares the three completed classifier baselines using existing metrics only. No training or LLM calls were run by this report generator.

| Model | Eval scope | Accuracy | Macro-F1 | Bug F1 | Feature F1 | Docs F1 | Question F1 | Latency ms/example | Size | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical TF-IDF + LogisticRegression | full test split (370 examples) | 0.7270 | 0.7123 | 0.6977 | 0.8586 | 0.7848 | 0.5079 | 0.3715 | 2.06 MB | n/a |
| Selected DistilBERT candidate | full test split (370 examples) | 0.6838 | 0.6649 | 0.5617 | 0.8571 | 0.8538 | 0.3871 | 7.8842 | 255.43 MB | n/a |
| LLM baseline (claude-haiku-4-5) | balanced sample (40 examples) | 0.6500 | 0.6010 | 0.5806 | 1.0000 | 0.8235 | 0.0000 | 1446.2508 | n/a | n/a |

## Recommendation

Deploy `classical_ml` for now.

Choose the classical baseline for now because it has the highest test macro-F1, the lowest latency, and a small local artifact. DistilBERT remains useful as a transformer baseline, and the LLM baseline is useful for audit comparison but is slower and weaker on the balanced sample.

## Notes

- Classical and DistilBERT results are measured on the full classification test split.
- The LLM baseline uses the Phase 6B balanced capped sample of 40 examples.
- LLM cost is null because token pricing was not configured for the run.

## Day 3 RAG eval results

The real RAG evaluation was run against the frozen 25-example golden set in `data/evals/rag_golden.jsonl`.

### Embedding comparison

| Embedding model | Hit@5 | MRR@10 |
|---|---:|---:|
| `bge-small-en-v1.5` | 0.36 | 0.218 |
| `all-MiniLM-L6-v2` | 0.32 | 0.180 |

Selected embedding model: `bge-small-en-v1.5`.

### Hybrid weighting

Selected weighting: dense 0.50 / sparse 0.50.

This weighting achieved the best Hit@5 at 0.36. Dense 0.75 / sparse 0.25 had a slightly higher MRR@10, but the difference was too small to outweigh the better Hit@5 from the balanced hybrid setting.

### Selected pipeline

Selected retrieval variant: `parent_child_hybrid`.

| Metric | Value |
|---|---:|
| Hit@5 | 0.36 |
| MRR@10 | 0.218 |
| Faithfulness | 4.52 |
| Answer relevancy | 4.28 |
| Judge pass rate | 0.44 |

### Interpretation

The first real RAG eval confirms that bounded Airflow docs plus held-out resolved issues can support measurable retrieval. `bge-small-en-v1.5` outperformed `all-MiniLM-L6-v2`, and balanced hybrid retrieval gave the best overall trade-off.

The judge pass rate is still modest, so the next improvement target is retrieval recall and context coverage rather than answer wording alone.