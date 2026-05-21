"""RAG evaluation harness (A11, DECISIONS D3.3 / D3.5 / D3.8 / D3.11 / D3.12).

Runs the EVALS.md variant ladder (naive+dense -> parent-child+dense -> +hybrid -> +rerank ->
+multi-query), the bge-vs-MiniLM embedding comparison, and the hybrid weight sweep over the RAG
golden set; computes Hit@5 / MRR@10 from chunk IDs; optionally generates answers (Claude Haiku)
and scores them with the frozen Sonnet judge rubric (strict JSON, fail-closed); writes
``artifacts/evals/rag_eval_report.json`` plus per-question retrieved-chunk snapshots.

All model/LLM dependencies are injected, so ``--mock`` runs the whole harness deterministically
with no downloads or API calls (used by tests/CI). Producing the real frozen numbers + setting
non-zero ``rag`` thresholds (D3.12) is the documented follow-up; until ``eval_thresholds.yaml``
gains a ``rag`` block this run is informational and does not gate.

Run:
    uv run --project ml python -m evals.rag_eval --mock           # deterministic smoke
    uv run --project ml python -m evals.rag_eval --no-generation  # real retrieval metrics only
    uv run --project ml python -m evals.rag_eval                  # full (needs models + API key)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
for _path in (REPO_ROOT, ML_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backend.app.domain.rag import (  # noqa: E402
    Chunk,
    ChunkLevel,
    GenerationMetrics,
    GoldenExample,
    RagEvalReport,
    RetrievalMetrics,
    RetrievalResult,
    RetrievedChunk,
    VariantConfig,
    VariantReport,
)
from rag.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    EmbeddingMatrix,
    get_embedder,
)
from rag.multiquery import expand_queries, generate_rewrites  # noqa: E402
from rag.rerank import DEFAULT_RERANK_TOP_K, rerank  # noqa: E402
from rag.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_PARAMS,
    WEIGHT_SWEEP,
    RetrievalIndex,
    RetrievalParams,
    build_retrieval_index,
)

JsonObject = dict[str, Any]

DEFAULT_ADVANCED_CHUNKS = REPO_ROOT / "data" / "processed" / "rag_chunks.jsonl"
DEFAULT_NAIVE_CHUNKS = REPO_ROOT / "data" / "processed" / "rag_chunks_naive.jsonl"
DEFAULT_GOLDEN = REPO_ROOT / "data" / "evals" / "rag_golden.jsonl"
DEFAULT_GOLDEN_DRAFT = REPO_ROOT / "data" / "evals" / "rag_golden.draft.jsonl"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "evals" / "rag_eval_report.json"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "artifacts" / "evals" / "rag_retrieved_snapshots"
DEFAULT_THRESHOLDS = REPO_ROOT / "eval_thresholds.yaml"
JUDGE_RUBRIC_PATH = REPO_ROOT / "backend" / "prompts" / "rag_judge_rubric_v1.md"
ANSWER_PROMPT_PATH = REPO_ROOT / "backend" / "prompts" / "rag_answer_v1.md"

GEN_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-4-6"
HIT_K = 5
MRR_K = 10
JUDGE_DIMENSIONS = (
    "faithfulness",
    "answer_relevancy",
    "context_usefulness",
    "completeness",
    "refusal_quality",
    "overall",
)
VALID_FAILURE_REASONS = frozenset(
    {
        "unsupported_claims",
        "context_contradiction",
        "ungrounded_generic",
        "confident_when_insufficient",
        "ignores_question",
        "relies_on_unretrieved",
    }
)


class Embedder(Protocol):
    def embed(
        self, texts: Sequence[str], *, is_query: bool = False
    ) -> EmbeddingMatrix: ...


EmbedderFactory = Callable[[str], Embedder]
GenerateFn = Callable[[str, str], str]
JudgeFn = Callable[[JsonObject], str]
RerankScoreFn = Callable[[str, Sequence[str]], list[float]]
RewriteFn = Callable[[str, int], str]


# --- I/O ---------------------------------------------------------------------


def read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    rows: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(record: JsonObject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(row) for row in read_jsonl(path)]


def load_golden(path: Path, draft_path: Path) -> tuple[list[GoldenExample], bool]:
    """Load the frozen golden set, falling back to the draft. Returns (examples, is_draft)."""
    if path.exists():
        return [GoldenExample.model_validate(row) for row in read_jsonl(path)], False
    return [GoldenExample.model_validate(row) for row in read_jsonl(draft_path)], True


def text_by_chunk_id(*chunk_lists: list[Chunk]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            mapping[chunk.chunk_id] = chunk.text
    return mapping


def children_of(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.level == ChunkLevel.CHILD]


# --- Retrieval metrics -------------------------------------------------------


def hit_at_k(retrieved_ids: Sequence[str], truth_ids: Sequence[str], k: int) -> float:
    truth = set(truth_ids)
    return 1.0 if truth & set(retrieved_ids[:k]) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], truth_ids: Sequence[str], k: int) -> float:
    truth = set(truth_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id in truth:
            return 1.0 / (rank + 1)
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --- Judge -------------------------------------------------------------------


def load_judge_rubric(path: Path = JUDGE_RUBRIC_PATH) -> str:
    return path.read_text(encoding="utf-8")


def failed_judge(reason: str) -> JsonObject:
    """A fail-closed verdict (all dimensions 1, pass false) for unusable judge output."""
    verdict: JsonObject = {dimension: 1 for dimension in JUDGE_DIMENSIONS}
    verdict["pass"] = False
    verdict["failure_reasons"] = [reason] if reason in VALID_FAILURE_REASONS else []
    verdict["short_rationale"] = f"judge output rejected: {reason}"
    return verdict


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def parse_judge_response(raw: str) -> JsonObject:
    """Parse strict judge JSON; fail closed (all-1, pass=false) on any malformation."""
    try:
        parsed = json.loads(_strip_code_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return failed_judge("invalid_json")
    if not isinstance(parsed, dict):
        return failed_judge("not_object")

    verdict: JsonObject = {}
    for dimension in JUDGE_DIMENSIONS:
        value = parsed.get(dimension)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not (1 <= value <= 5)
        ):
            return failed_judge("invalid_score")
        verdict[dimension] = value
    if not isinstance(parsed.get("pass"), bool):
        return failed_judge("invalid_pass")
    reasons = parsed.get("failure_reasons", [])
    if not isinstance(reasons, list) or any(
        r not in VALID_FAILURE_REASONS for r in reasons
    ):
        return failed_judge("invalid_failure_reasons")
    verdict["pass"] = parsed["pass"]
    verdict["failure_reasons"] = reasons
    verdict["short_rationale"] = str(parsed.get("short_rationale", ""))
    return verdict


def build_judge_payload(
    gold: GoldenExample,
    result: RetrievalResult,
    answer: str,
    context: str,
) -> JsonObject:
    return {
        "question": gold.question,
        "retrieved_context": context,
        "generated_answer": answer,
        "ideal_answer": gold.ideal_answer,
        "ground_truth_chunk_ids": list(gold.ground_truth_child_chunk_ids),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in result.retrieved],
    }


def aggregate_generation(verdicts: list[JsonObject]) -> GenerationMetrics:
    return GenerationMetrics(
        faithfulness=_mean([float(v["faithfulness"]) for v in verdicts]),
        answer_relevancy=_mean([float(v["answer_relevancy"]) for v in verdicts]),
        context_usefulness=_mean([float(v["context_usefulness"]) for v in verdicts]),
        completeness=_mean([float(v["completeness"]) for v in verdicts]),
        refusal_quality=_mean([float(v["refusal_quality"]) for v in verdicts]),
        overall=_mean([float(v["overall"]) for v in verdicts]),
        pass_rate=_mean([1.0 if v["pass"] else 0.0 for v in verdicts]),
        num_examples=len(verdicts),
    )


# --- Generation context ------------------------------------------------------


def build_context(parent_ids: Sequence[str], text_by_id: dict[str, str]) -> str:
    blocks = [text_by_id[pid] for pid in parent_ids if pid in text_by_id]
    return "\n\n---\n\n".join(blocks)


def load_answer_prompt(path: Path = ANSWER_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


# --- Variants ----------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    name: str
    config: VariantConfig
    group: str  # "ladder" | "embedding_comparison" | "weight_sweep"


def variant_ladder(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> list[VariantSpec]:
    return [
        VariantSpec(
            "naive_dense",
            VariantConfig(embedding_model=embedding_model, chunking="naive_fixed"),
            "ladder",
        ),
        VariantSpec(
            "parent_child_dense",
            VariantConfig(embedding_model=embedding_model, chunking="parent_child"),
            "ladder",
        ),
        VariantSpec(
            "parent_child_hybrid",
            VariantConfig(
                embedding_model=embedding_model,
                chunking="parent_child",
                use_hybrid=True,
                dense_weight=0.5,
                sparse_weight=0.5,
            ),
            "ladder",
        ),
        VariantSpec(
            "parent_child_hybrid_rerank",
            VariantConfig(
                embedding_model=embedding_model,
                chunking="parent_child",
                use_hybrid=True,
                dense_weight=0.5,
                sparse_weight=0.5,
                use_rerank=True,
                rerank_top_k=DEFAULT_RERANK_TOP_K,
            ),
            "ladder",
        ),
        VariantSpec(
            "parent_child_hybrid_rerank_multiquery",
            VariantConfig(
                embedding_model=embedding_model,
                chunking="parent_child",
                use_hybrid=True,
                dense_weight=0.5,
                sparse_weight=0.5,
                use_rerank=True,
                rerank_top_k=DEFAULT_RERANK_TOP_K,
                multi_query=3,
            ),
            "ladder",
        ),
    ]


def embedding_comparison_specs() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for model_key in sorted(EMBEDDING_MODELS):
        specs.append(
            VariantSpec(
                f"hybrid_{model_key}",
                VariantConfig(
                    embedding_model=model_key,
                    chunking="parent_child",
                    use_hybrid=True,
                    dense_weight=0.5,
                    sparse_weight=0.5,
                ),
                "embedding_comparison",
            )
        )
    return specs


def weight_sweep_specs(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for dense_weight, sparse_weight in WEIGHT_SWEEP:
        specs.append(
            VariantSpec(
                f"hybrid_dense{dense_weight}_sparse{sparse_weight}",
                VariantConfig(
                    embedding_model=embedding_model,
                    chunking="parent_child",
                    use_hybrid=True,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                ),
                "weight_sweep",
            )
        )
    return specs


# --- Pipeline ----------------------------------------------------------------


def _result_from_chunks(
    query: str, chunks: list[RetrievedChunk], params: RetrievalParams
) -> RetrievalResult:
    ranked = [
        RetrievedChunk(
            chunk_id=c.chunk_id, parent_id=c.parent_id, score=c.score, rank=rank
        )
        for rank, c in enumerate(chunks)
    ]
    context_parent_ids: list[str] = []
    for chunk in ranked[: params.final_context_chunks]:
        parent = chunk.parent_id or chunk.chunk_id
        if parent not in context_parent_ids:
            context_parent_ids.append(parent)
    return RetrievalResult(
        query=query, retrieved=ranked, context_parent_ids=context_parent_ids
    )


def _fuse(results: list[RetrievalResult], params: RetrievalParams) -> RetrievalResult:
    best: dict[str, RetrievedChunk] = {}
    for result in results:
        for chunk in result.retrieved:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    ordered = sorted(best.values(), key=lambda c: (-c.score, c.chunk_id))[
        : params.hybrid_top_k
    ]
    return _result_from_chunks(results[0].query if results else "", ordered, params)


def run_pipeline(
    question: str,
    config: VariantConfig,
    index: RetrievalIndex,
    embedder: Embedder,
    text_by_id: dict[str, str],
    *,
    rerank_score_fn: RerankScoreFn | None,
    rewrite_fn: RewriteFn | None,
    params: RetrievalParams = DEFAULT_RETRIEVAL_PARAMS,
) -> RetrievalResult:
    queries = [question]
    if config.multi_query and rewrite_fn is not None:
        rewrites = generate_rewrites(question, rewrite_fn, n=config.multi_query)
        queries = expand_queries(question, rewrites)

    per_query: list[RetrievalResult] = []
    vectors = embedder.embed(queries, is_query=True)
    for query, vector in zip(queries, vectors, strict=True):
        if config.use_hybrid:
            per_query.append(
                index.search_hybrid(
                    vector,
                    query,
                    dense_weight=config.dense_weight or 0.5,
                    sparse_weight=config.sparse_weight or 0.5,
                    params=params,
                )
            )
        else:
            per_query.append(index.search_dense(vector, query, params))

    result = per_query[0] if len(per_query) == 1 else _fuse(per_query, params)

    if config.use_rerank and rerank_score_fn is not None:
        reranked = rerank(
            question,
            result.retrieved,
            text_by_id,
            rerank_score_fn,
            top_k=config.rerank_top_k or DEFAULT_RERANK_TOP_K,
        )
        result = _result_from_chunks(question, reranked, params)
    return result


def run_variant(
    spec: VariantSpec,
    goldens: list[GoldenExample],
    *,
    children: list[Chunk],
    embedder: Embedder,
    text_by_id: dict[str, str],
    rerank_score_fn: RerankScoreFn | None,
    rewrite_fn: RewriteFn | None,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    params: RetrievalParams = DEFAULT_RETRIEVAL_PARAMS,
) -> tuple[VariantReport, list[JsonObject]]:
    embeddings = embedder.embed([chunk.text for chunk in children])
    index = build_retrieval_index(children, embeddings)

    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    verdicts: list[JsonObject] = []
    snapshots: list[JsonObject] = []

    for gold in goldens:
        result = run_pipeline(
            gold.question,
            spec.config,
            index,
            embedder,
            text_by_id,
            rerank_score_fn=rerank_score_fn,
            rewrite_fn=rewrite_fn,
            params=params,
        )
        retrieved_ids = [chunk.chunk_id for chunk in result.retrieved]
        hits.append(hit_at_k(retrieved_ids, gold.ground_truth_child_chunk_ids, HIT_K))
        reciprocal_ranks.append(
            mrr_at_k(retrieved_ids, gold.ground_truth_child_chunk_ids, MRR_K)
        )
        snapshots.append(
            {
                "variant": spec.name,
                "question": gold.question,
                "retrieved_chunk_ids": retrieved_ids[: params.final_context_chunks],
                "context_parent_ids": result.context_parent_ids,
                "ground_truth_child_chunk_ids": list(gold.ground_truth_child_chunk_ids),
            }
        )
        if generate_fn is not None and judge_fn is not None:
            context = build_context(result.context_parent_ids, text_by_id)
            answer = generate_fn(gold.question, context)
            payload = build_judge_payload(gold, result, answer, context)
            verdicts.append(parse_judge_response(judge_fn(payload)))

    retrieval = RetrievalMetrics(
        hit_at_5=_mean(hits),
        mrr_at_10=_mean(reciprocal_ranks),
        num_examples=len(goldens),
    )
    generation = aggregate_generation(verdicts) if verdicts else None
    report = VariantReport(
        name=spec.name, config=spec.config, retrieval=retrieval, generation=generation
    )
    return report, snapshots


def run_eval(
    specs: list[VariantSpec],
    goldens: list[GoldenExample],
    advanced_children: list[Chunk],
    naive_children: list[Chunk],
    text_by_id: dict[str, str],
    *,
    embedder_factory: EmbedderFactory,
    rerank_score_fn: RerankScoreFn | None,
    rewrite_fn: RewriteFn | None,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    notes: list[str] | None = None,
) -> tuple[RagEvalReport, list[JsonObject]]:
    grouped: dict[str, list[VariantReport]] = {
        "ladder": [],
        "embedding_comparison": [],
        "weight_sweep": [],
    }
    all_snapshots: list[JsonObject] = []
    for spec in specs:
        children = (
            naive_children
            if spec.config.chunking == "naive_fixed"
            else advanced_children
        )
        embedder = embedder_factory(spec.config.embedding_model)
        report, snapshots = run_variant(
            spec,
            goldens,
            children=children,
            embedder=embedder,
            text_by_id=text_by_id,
            rerank_score_fn=rerank_score_fn,
            rewrite_fn=rewrite_fn,
            generate_fn=generate_fn,
            judge_fn=judge_fn,
        )
        grouped.setdefault(spec.group, []).append(report)
        all_snapshots.extend(snapshots)

    eval_report = RagEvalReport(
        generated_at=datetime.now(UTC),
        variants=grouped["ladder"],
        embedding_comparison=grouped["embedding_comparison"],
        weight_sweep=grouped["weight_sweep"],
        notes=notes or [],
    )
    return eval_report, all_snapshots


def write_snapshots(snapshots: list[JsonObject], snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / "rag_retrieved_snapshots.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshots, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# --- Thresholds (optional until first baseline, D3.12) -----------------------


def load_rag_thresholds(path: Path) -> JsonObject | None:
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    rag = parsed.get("rag") if isinstance(parsed, dict) else None
    return rag if isinstance(rag, dict) else None


# --- Real model/LLM functions (not used in --mock or tests) ------------------


def real_embedder_factory(model_key: str) -> Embedder:
    return get_embedder(model_key)


def anthropic_generate_fn(question: str, context: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        load_answer_prompt()
        .replace("{question}", question)
        .replace("{context}", context)
    )
    message = client.messages.create(
        model=GEN_MODEL, max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def anthropic_judge_fn(payload: JsonObject) -> str:
    import anthropic

    client = anthropic.Anthropic()
    rubric = load_judge_rubric()
    content = f"{rubric}\n\n## Inputs to score\n\n```json\n{json.dumps(payload, indent=2)}\n```"
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def anthropic_rewrite_fn(query: str, n: int) -> str:
    from rag.multiquery import anthropic_rewrite_fn as _rewrite

    return _rewrite(query, n)


# --- Mock functions (deterministic, offline) ---------------------------------


class MockEmbedder:
    """Deterministic hash-based unit-vector embedder for offline/mock runs."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> EmbeddingMatrix:
        vectors = [self._vector(text) for text in texts]
        return (
            np.stack(vectors).astype(np.float32)
            if vectors
            else np.zeros((0, self.dim), np.float32)
        )

    def _vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        generator = np.random.default_rng(seed)
        vector = generator.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


def mock_embedder_factory(model_key: str) -> Embedder:
    return MockEmbedder()


def mock_rerank_score_fn(query: str, passages: Sequence[str]) -> list[float]:
    query_tokens = set(query.lower().split())
    return [
        float(len(query_tokens & set(passage.lower().split()))) for passage in passages
    ]


def mock_generate_fn(question: str, context: str) -> str:
    return f"Based on the retrieved context: {context[:200]}"


def mock_rewrite_fn(query: str, n: int) -> str:
    return "\n".join(f"{query} variant {index}" for index in range(n))


def mock_judge_fn(payload: JsonObject) -> str:
    grounded = bool(payload.get("retrieved_context"))
    score = 5 if grounded else 2
    return json.dumps(
        {
            "faithfulness": score,
            "answer_relevancy": score,
            "context_usefulness": score,
            "completeness": score,
            "refusal_quality": 5,
            "overall": score,
            "pass": grounded,
            "failure_reasons": [],
            "short_rationale": "mock verdict",
        }
    )


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument(
        "--mock", action="store_true", help="Use deterministic offline stand-ins."
    )
    parser.add_argument(
        "--no-generation", action="store_true", help="Skip answer generation + judging."
    )
    parser.add_argument("--advanced-chunks", type=Path, default=DEFAULT_ADVANCED_CHUNKS)
    parser.add_argument("--naive-chunks", type=Path, default=DEFAULT_NAIVE_CHUNKS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    advanced = load_chunks(args.advanced_chunks)
    naive = load_chunks(args.naive_chunks)
    goldens, is_draft = load_golden(args.golden, DEFAULT_GOLDEN_DRAFT)
    if not advanced or not goldens:
        print(
            "ERROR: missing chunk artifacts or golden set; run build_chunks + golden first."
        )
        return 1

    text_by_id = text_by_chunk_id(advanced, naive)
    advanced_children = children_of(advanced)
    naive_children = children_of(naive)

    specs = variant_ladder() + embedding_comparison_specs() + weight_sweep_specs()
    notes: list[str] = []
    if is_draft:
        notes.append(
            "Golden set is the DRAFT (human_reviewed=false); numbers are provisional."
        )

    if args.mock:
        embedder_factory: EmbedderFactory = mock_embedder_factory
        rerank_score_fn: RerankScoreFn | None = mock_rerank_score_fn
        rewrite_fn: RewriteFn | None = mock_rewrite_fn
        generate_fn: GenerateFn | None = (
            None if args.no_generation else mock_generate_fn
        )
        judge_fn: JudgeFn | None = None if args.no_generation else mock_judge_fn
        notes.append("MOCK run: deterministic stand-ins, not real model/LLM numbers.")
    else:
        embedder_factory = real_embedder_factory
        rerank_score_fn = get_reranker_score_fn()
        rewrite_fn = anthropic_rewrite_fn
        generate_fn = None if args.no_generation else anthropic_generate_fn
        judge_fn = None if args.no_generation else anthropic_judge_fn

    report, snapshots = run_eval(
        specs,
        goldens,
        advanced_children,
        naive_children,
        text_by_id,
        embedder_factory=embedder_factory,
        rerank_score_fn=rerank_score_fn,
        rewrite_fn=rewrite_fn,
        generate_fn=generate_fn,
        judge_fn=judge_fn,
        notes=notes,
    )

    write_json(report.model_dump(mode="json"), args.report)
    write_snapshots(snapshots[:50], args.snapshot_dir)

    for note in notes:
        print(f"NOTE: {note}")
    for variant in report.variants:
        gen = variant.generation
        gen_str = f" | pass_rate={gen.pass_rate:.2f}" if gen else ""
        print(
            f"{variant.name:42s} hit@5={variant.retrieval.hit_at_5:.3f} "
            f"mrr@10={variant.retrieval.mrr_at_10:.3f}{gen_str}"
        )

    thresholds = load_rag_thresholds(args.thresholds)
    if thresholds is None:
        print(
            "NOTE: no `rag` thresholds set yet (D3.12); informational run, not gating."
        )
        return 0
    return gate_against_thresholds(report, thresholds)


def get_reranker_score_fn() -> RerankScoreFn:
    from rag.rerank import get_reranker

    return get_reranker().score


def gate_against_thresholds(report: RagEvalReport, thresholds: JsonObject) -> int:
    """Gate the best ladder variant against committed non-zero rag thresholds."""
    if not report.variants:
        return 1
    best = max(report.variants, key=lambda v: v.retrieval.hit_at_5)
    hit_ok = best.retrieval.hit_at_5 >= float(thresholds.get("hit_at_5", 0.0))
    mrr_ok = best.retrieval.mrr_at_10 >= float(thresholds.get("mrr_at_10", 0.0))
    passed = hit_ok and mrr_ok
    print(f"threshold gate: {'PASS' if passed else 'FAIL'} (best variant {best.name})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
