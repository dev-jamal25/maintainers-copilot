from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = REPO_ROOT / "ml"
for _path in (REPO_ROOT, ML_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backend.app.domain.rag import (  # noqa: E402
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkSourceType,
    Difficulty,
    GoldenExample,
    GoldenSourceType,
    SourceRef,
)
from evals.rag_eval import (  # noqa: E402
    MockEmbedder,
    aggregate_generation,
    embedding_comparison_specs,
    estimate_live_llm_calls,
    failed_judge,
    gate_against_thresholds,
    hit_at_k,
    mock_embedder_factory,
    mock_generate_fn,
    mock_judge_fn,
    mock_rerank_score_fn,
    mock_rewrite_fn,
    mrr_at_k,
    parse_judge_response,
    run_eval,
    run_variant,
    variant_ladder,
    weight_sweep_specs,
)


def _doc_chunks() -> list[Chunk]:
    parent = Chunk(
        level=ChunkLevel.PARENT,
        text="alpha beta gamma delta epsilon",
        metadata=ChunkMetadata(
            chunk_id="doc:x:parent:0",
            parent_id=None,
            source_type=ChunkSourceType.DOCS,
            source_id="x",
        ),
    )
    child0 = Chunk(
        level=ChunkLevel.CHILD,
        text="alpha beta gamma",
        metadata=ChunkMetadata(
            chunk_id="doc:x:parent:0:child:0",
            parent_id="doc:x:parent:0",
            source_type=ChunkSourceType.DOCS,
            source_id="x",
        ),
    )
    child1 = Chunk(
        level=ChunkLevel.CHILD,
        text="zeta eta theta",
        metadata=ChunkMetadata(
            chunk_id="doc:x:parent:0:child:1",
            parent_id="doc:x:parent:0",
            source_type=ChunkSourceType.DOCS,
            source_id="x",
        ),
    )
    return [parent, child0, child1]


def _naive_chunks() -> list[Chunk]:
    return [
        Chunk(
            level=ChunkLevel.CHILD,
            text="alpha beta gamma",
            metadata=ChunkMetadata(
                chunk_id="doc:x:naive:0", source_type=ChunkSourceType.DOCS, source_id="x"
            ),
        )
    ]


def _golden() -> GoldenExample:
    return GoldenExample(
        question="alpha beta gamma",
        ideal_answer="The grounded answer mentions alpha beta gamma in enough detail to pass.",
        ground_truth_sources=[
            SourceRef(source_type=ChunkSourceType.DOCS, source_id="x", section_path=["X"])
        ],
        difficulty=Difficulty.EASY,
        source_type=GoldenSourceType.DOCS_ONLY,
        ground_truth_parent_ids=["doc:x:parent:0"],
        ground_truth_child_chunk_ids=["doc:x:parent:0:child:0"],
    )


def test_hit_and_mrr() -> None:
    assert hit_at_k(["a", "b", "c"], ["c"], 5) == 1.0
    assert hit_at_k(["a", "b"], ["z"], 5) == 0.0
    assert mrr_at_k(["a", "b", "c"], ["b"], 10) == 0.5
    assert mrr_at_k(["a", "b"], ["z"], 10) == 0.0


def test_mock_embedder_is_deterministic_and_normalized() -> None:
    embedder = MockEmbedder(dim=16)
    first = embedder.embed(["scheduler"])
    second = embedder.embed(["scheduler"])
    assert (first == second).all()
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5


def test_parse_judge_valid_and_failclosed() -> None:
    valid = json.dumps(
        {
            "faithfulness": 5,
            "answer_relevancy": 4,
            "context_usefulness": 4,
            "completeness": 4,
            "refusal_quality": 5,
            "overall": 4,
            "pass": True,
            "failure_reasons": [],
            "short_rationale": "ok",
        }
    )
    verdict = parse_judge_response(valid)
    assert verdict["pass"] is True
    assert verdict["faithfulness"] == 5

    assert parse_judge_response("not json at all")["pass"] is False
    assert parse_judge_response('{"faithfulness": 9}')["pass"] is False  # out of range
    fenced = f"```json\n{valid}\n```"
    assert parse_judge_response(fenced)["pass"] is True  # fences stripped


def test_failed_judge_shape() -> None:
    verdict = failed_judge("invalid_json")
    assert verdict["pass"] is False
    assert verdict["overall"] == 1


def test_aggregate_generation_means() -> None:
    verdicts = [
        {**failed_judge("invalid_json"), "pass": True},
        failed_judge("invalid_json"),
    ]
    metrics = aggregate_generation(verdicts)
    assert metrics.num_examples == 2
    assert metrics.pass_rate == 0.5


def test_run_variant_retrieval_hits_known_chunk() -> None:
    spec = variant_ladder()[1]  # parent_child_dense
    report, snapshots = run_variant(
        spec,
        [_golden()],
        children=[c for c in _doc_chunks() if c.level == ChunkLevel.CHILD],
        embedder=MockEmbedder(),
        text_by_id={c.chunk_id: c.text for c in _doc_chunks()},
        rerank_score_fn=None,
        rewrite_fn=None,
        generate_fn=None,
        judge_fn=None,
    )
    assert report.retrieval.hit_at_5 == 1.0
    assert report.retrieval.mrr_at_10 == 1.0
    assert report.generation is None
    assert snapshots[0]["variant"] == "parent_child_dense"


def test_run_variant_with_generation_and_judge() -> None:
    spec = variant_ladder()[1]
    report, _ = run_variant(
        spec,
        [_golden()],
        children=[c for c in _doc_chunks() if c.level == ChunkLevel.CHILD],
        embedder=MockEmbedder(),
        text_by_id={c.chunk_id: c.text for c in _doc_chunks()},
        rerank_score_fn=mock_rerank_score_fn,
        rewrite_fn=mock_rewrite_fn,
        generate_fn=mock_generate_fn,
        judge_fn=mock_judge_fn,
    )
    assert report.generation is not None
    assert report.generation.pass_rate == 1.0


def test_run_eval_groups_variants() -> None:
    specs = variant_ladder()[:2] + embedding_comparison_specs() + weight_sweep_specs()
    report, snapshots = run_eval(
        specs,
        [_golden()],
        advanced_children=[c for c in _doc_chunks() if c.level == ChunkLevel.CHILD],
        naive_children=_naive_chunks(),
        text_by_id={c.chunk_id: c.text for c in _doc_chunks()},
        embedder_factory=mock_embedder_factory,
        rerank_score_fn=mock_rerank_score_fn,
        rewrite_fn=mock_rewrite_fn,
        generate_fn=None,
        judge_fn=None,
    )
    assert len(report.variants) == 2
    assert len(report.embedding_comparison) == 2
    assert len(report.weight_sweep) == 3
    assert snapshots


def test_spec_counts_match_decisions() -> None:
    assert len(variant_ladder()) == 5
    assert len(embedding_comparison_specs()) == 2
    assert len(weight_sweep_specs()) == 3


def test_estimate_live_llm_calls_with_generation() -> None:
    specs = variant_ladder() + embedding_comparison_specs() + weight_sweep_specs()
    # 10 variants total, exactly one multi-query variant (the full ladder tail).
    estimate = estimate_live_llm_calls(specs, 3, generation_enabled=True)
    assert estimate["generation"] == 10 * 3
    assert estimate["judge"] == 10 * 3
    assert estimate["rewrite"] == 1 * 3
    assert estimate["total"] == 30 + 30 + 3


def test_estimate_live_llm_calls_retrieval_only() -> None:
    specs = variant_ladder() + embedding_comparison_specs() + weight_sweep_specs()
    # Generation disabled: no generation/judge calls, only multi-query rewrites remain.
    estimate = estimate_live_llm_calls(specs, 3, generation_enabled=False)
    assert estimate["generation"] == 0
    assert estimate["judge"] == 0
    assert estimate["rewrite"] == 1 * 3
    assert estimate["total"] == 3


def test_gate_against_thresholds() -> None:
    report, _ = run_eval(
        variant_ladder()[:1],
        [_golden()],
        advanced_children=[c for c in _doc_chunks() if c.level == ChunkLevel.CHILD],
        naive_children=_naive_chunks(),
        text_by_id={c.chunk_id: c.text for c in _doc_chunks()},
        embedder_factory=mock_embedder_factory,
        rerank_score_fn=None,
        rewrite_fn=None,
        generate_fn=None,
        judge_fn=None,
    )
    assert gate_against_thresholds(report, {"hit_at_5": 0.0, "mrr_at_10": 0.0}) == 0
    assert gate_against_thresholds(report, {"hit_at_5": 1.5, "mrr_at_10": 0.0}) == 1
