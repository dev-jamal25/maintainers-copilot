from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classification.data import LABEL_ORDER, REPO_ROOT, display_path, resolve_repo_path

JsonObject = dict[str, Any]

DEFAULT_CLASSICAL_METRICS_PATH = (
    REPO_ROOT / "artifacts" / "classification" / "classical" / "test_metrics.json"
)
DEFAULT_DISTILBERT_MANIFEST_PATH = (
    REPO_ROOT / "artifacts" / "classification" / "distilbert_selected_manifest.json"
)
DEFAULT_LLM_METRICS_PATH = (
    REPO_ROOT / "artifacts" / "classification" / "llm_baseline" / "metrics.json"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "classification" / "model_comparison.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "deliverables" / "model_comparison.md"
DEFAULT_DECISIONS_PATH = REPO_ROOT / "deliverables" / "DECISIONS.md"

CLASSICAL_MODEL_NAME = "classical_ml"
DISTILBERT_MODEL_NAME = "selected_distilbert"
LLM_MODEL_NAME = "llm_baseline"
DEPLOYMENT_CHOICE = CLASSICAL_MODEL_NAME


@dataclass(frozen=True)
class ComparisonPaths:
    classical_metrics_path: Path = DEFAULT_CLASSICAL_METRICS_PATH
    distilbert_manifest_path: Path = DEFAULT_DISTILBERT_MANIFEST_PATH
    llm_metrics_path: Path = DEFAULT_LLM_METRICS_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    decisions_path: Path = DEFAULT_DECISIONS_PATH


def read_json(path: Path) -> JsonObject:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        parsed: object = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{display_path(resolved)} must contain a JSON object")
    return parsed


def write_json(record: JsonObject, path: Path) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_text(text: str, path: Path) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def metric_float(metrics: JsonObject, key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def per_class_f1(metrics: JsonObject) -> dict[str, float | None]:
    values = metrics.get("per_class_f1", {})
    if not isinstance(values, dict):
        values = {}
    return {
        label: float(values[label])
        if isinstance(values.get(label), int | float) and not isinstance(values.get(label), bool)
        else None
        for label in LABEL_ORDER
    }


def latency_ms(metrics: JsonObject, *, fallback_key: str | None = None) -> float | None:
    latency = metrics.get("latency")
    if isinstance(latency, dict):
        value = latency.get("ms_per_example")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    if fallback_key is not None:
        return metric_float(metrics, fallback_key)
    return None


def file_size(path: Path | None) -> int | None:
    if path is None:
        return None
    resolved = resolve_repo_path(path)
    return resolved.stat().st_size if resolved.exists() else None


def artifact_path_from_model_metrics(metrics: JsonObject) -> Path | None:
    model = metrics.get("model", {})
    if not isinstance(model, dict):
        return None
    raw_path = model.get("artifact_path") or model.get("path")
    return Path(str(raw_path)) if raw_path else None


def evaluation_scope(*, examples: int | None, sampling_mode: str | None = None) -> str:
    if sampling_mode == "limit_per_class":
        return f"balanced sample ({examples} examples)"
    if examples is None:
        return "unknown"
    return f"full test split ({examples} examples)"


def row_common(
    *,
    name: str,
    display_name: str,
    accuracy: float | None,
    macro_f1: float | None,
    per_class: dict[str, float | None],
    latency_ms_per_example: float | None,
    model_size_bytes: int | None,
    estimated_cost_usd: float | None,
    evaluation_examples: int | None,
    evaluation_scope_text: str,
    notes: list[str],
) -> JsonObject:
    return {
        "accuracy": accuracy,
        "display_name": display_name,
        "estimated_cost_usd": estimated_cost_usd,
        "evaluation_examples": evaluation_examples,
        "evaluation_scope": evaluation_scope_text,
        "latency_ms_per_example": latency_ms_per_example,
        "macro_f1": macro_f1,
        "model_size_bytes": model_size_bytes,
        "name": name,
        "notes": notes,
        "per_class_f1": per_class,
    }


def classical_row(metrics: JsonObject) -> JsonObject:
    latency = metrics.get("latency", {})
    examples = latency.get("examples") if isinstance(latency, dict) else None
    model = metrics.get("model", {})
    size = model.get("size_bytes") if isinstance(model, dict) else None
    return row_common(
        name=CLASSICAL_MODEL_NAME,
        display_name="Classical TF-IDF + LogisticRegression",
        accuracy=metric_float(metrics, "accuracy"),
        macro_f1=metric_float(metrics, "macro_f1"),
        per_class=per_class_f1(metrics),
        latency_ms_per_example=latency_ms(metrics),
        model_size_bytes=int(size) if isinstance(size, int) else None,
        estimated_cost_usd=None,
        evaluation_examples=int(examples) if isinstance(examples, int) else None,
        evaluation_scope_text=evaluation_scope(
            examples=int(examples) if isinstance(examples, int) else None
        ),
        notes=["Best macro-F1 and fastest latency among available baselines."],
    )


def distilbert_row(manifest: JsonObject) -> JsonObject:
    selected_model_dir = Path(str(manifest["selected_model_dir"]))
    artifact_path = selected_model_dir / "model.safetensors"
    if not resolve_repo_path(artifact_path).exists():
        artifact_path = selected_model_dir / "pytorch_model.bin"
    test_metrics_path = resolve_repo_path(selected_model_dir / "test_metrics.json")
    test_metrics = read_json(test_metrics_path) if test_metrics_path.exists() else {}
    latency = test_metrics.get("latency", {})
    examples = latency.get("examples") if isinstance(latency, dict) else None
    evaluation_examples = int(examples) if isinstance(examples, int) else None
    return row_common(
        name=DISTILBERT_MODEL_NAME,
        display_name="Selected DistilBERT candidate",
        accuracy=metric_float(manifest, "test_accuracy"),
        macro_f1=metric_float(manifest, "test_macro_f1"),
        per_class=per_class_f1(manifest),
        latency_ms_per_example=metric_float(manifest, "latency_ms_per_example"),
        model_size_bytes=file_size(artifact_path),
        estimated_cost_usd=None,
        evaluation_examples=evaluation_examples,
        evaluation_scope_text=evaluation_scope(examples=evaluation_examples),
        notes=[
            "Selected via Phase 5D manifest.",
            "Improves prior DistilBERT, but trails the classical baseline on macro-F1.",
        ],
    )


def llm_row(metrics: JsonObject) -> JsonObject:
    examples = metrics.get("total_examples_attempted")
    sampling_mode = metrics.get("sampling_mode")
    return row_common(
        name=LLM_MODEL_NAME,
        display_name=f"LLM baseline ({metrics.get('model', 'unknown')})",
        accuracy=metric_float(metrics, "accuracy"),
        macro_f1=metric_float(metrics, "macro_f1"),
        per_class=per_class_f1(metrics),
        latency_ms_per_example=latency_ms(metrics, fallback_key="average_latency_ms"),
        model_size_bytes=None,
        estimated_cost_usd=metric_float(metrics, "estimated_cost_usd"),
        evaluation_examples=int(examples) if isinstance(examples, int) else None,
        evaluation_scope_text=evaluation_scope(
            examples=int(examples) if isinstance(examples, int) else None,
            sampling_mode=str(sampling_mode) if isinstance(sampling_mode, str) else None,
        ),
        notes=[
            "Evaluated on the Phase 6B balanced capped sample, not the full test split.",
            "Question F1 is currently 0.0 on the balanced sample.",
        ],
    ) | {
        "sampling_mode": sampling_mode,
        "selected_per_class": metrics.get("selected_per_class"),
        "token_usage": metrics.get("token_usage"),
    }


def macro_f1_value(row: JsonObject) -> float:
    value = row.get("macro_f1")
    return float(value) if isinstance(value, int | float) else 0.0


def latency_value(row: JsonObject) -> float:
    value = row.get("latency_ms_per_example")
    return float(value) if isinstance(value, int | float) else float("inf")


def select_deployment_model(rows: list[JsonObject]) -> JsonObject:
    candidates = [row for row in rows if row["name"] == DEPLOYMENT_CHOICE]
    if candidates:
        return candidates[0]
    return sorted(rows, key=lambda row: (-macro_f1_value(row), latency_value(row)))[0]


def comparison_payload(
    *,
    classical_metrics: JsonObject,
    distilbert_manifest: JsonObject,
    llm_metrics: JsonObject,
    paths: ComparisonPaths,
) -> JsonObject:
    rows = [
        classical_row(classical_metrics),
        distilbert_row(distilbert_manifest),
        llm_row(llm_metrics),
    ]
    recommendation = select_deployment_model(rows)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "classical_metrics": display_path(resolve_repo_path(paths.classical_metrics_path)),
            "distilbert_selected_manifest": display_path(
                resolve_repo_path(paths.distilbert_manifest_path)
            ),
            "llm_metrics": display_path(resolve_repo_path(paths.llm_metrics_path)),
        },
        "models": rows,
        "recommendation": {
            "deployment_choice": recommendation["name"],
            "reason": (
                "Choose the classical baseline for now because it has the highest test "
                "macro-F1, the lowest latency, and a small local artifact. DistilBERT remains "
                "useful as a transformer baseline, and the LLM baseline is useful for audit "
                "comparison but is slower and weaker on the balanced sample."
            ),
        },
        "selection_rule": {
            "primary": "prefer highest macro-F1 on the relevant held-out evaluation",
            "secondary": "prefer lower latency and lower operational cost when metrics are close",
            "scope_note": (
                "Classical and DistilBERT metrics are full test split results. "
                "LLM metrics are from the Phase 6B balanced capped sample."
            ),
        },
    }


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def size_mb(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value) / (1024 * 1024):.2f} MB"
    return "n/a"


def markdown_table(rows: list[JsonObject]) -> str:
    header = (
        "| Model | Eval scope | Accuracy | Macro-F1 | Bug F1 | Feature F1 | Docs F1 | "
        "Question F1 | Latency ms/example | Size | Cost |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        per_class = row["per_class_f1"]
        cost = row.get("estimated_cost_usd")
        lines.append(
            "| "
            f"{row['display_name']} | "
            f"{row['evaluation_scope']} | "
            f"{fmt(row['accuracy'])} | "
            f"{fmt(row['macro_f1'])} | "
            f"{fmt(per_class['bug'])} | "
            f"{fmt(per_class['feature'])} | "
            f"{fmt(per_class['docs'])} | "
            f"{fmt(per_class['question'])} | "
            f"{fmt(row['latency_ms_per_example'])} | "
            f"{size_mb(row['model_size_bytes'])} | "
            f"{fmt(cost, digits=6)} |"
        )
    return "\n".join(lines)


def render_markdown(comparison: JsonObject) -> str:
    rows = comparison["models"]
    recommendation = comparison["recommendation"]
    return (
        "# Classification Model Comparison\n\n"
        "Phase 7 compares the three completed classifier baselines using existing metrics only. "
        "No training or LLM calls were run by this report generator.\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Recommendation\n\n"
        f"Deploy `{recommendation['deployment_choice']}` for now.\n\n"
        f"{recommendation['reason']}\n\n"
        "## Notes\n\n"
        "- Classical and DistilBERT results are measured on the full classification test split.\n"
        "- The LLM baseline uses the Phase 6B balanced capped sample of 40 examples.\n"
        "- LLM cost is null because token pricing was not configured for the run.\n"
    )


def decision_entry(comparison: JsonObject) -> str:
    recommendation = comparison["recommendation"]
    return (
        "## D2.2 Classification deployment candidate: classical baseline\n\n"
        "**Decision:** Use the classical TF-IDF + LogisticRegression baseline as the current "
        "deployment candidate for issue classification.\n\n"
        "**Reasoning:**\n\n"
        "- It has the strongest full-test macro-F1 among the completed local baselines.\n"
        "- It is far faster than the selected DistilBERT candidate and the LLM baseline.\n"
        "- It has a small local artifact and no per-request inference cost.\n"
        "- The selected DistilBERT candidate remains useful for transformer comparison, but it "
        "does not beat the classical baseline yet.\n"
        "- The LLM baseline remains useful as an audit/comparison baseline, but the Phase 6B "
        "balanced sample is slower and weaker, especially on `question`.\n\n"
        f"**Recommendation summary:** {recommendation['reason']}\n"
    )


def upsert_decision_entry(path: Path, entry: str) -> None:
    resolved = resolve_repo_path(path)
    if resolved.exists():
        content = resolved.read_text(encoding="utf-8")
    else:
        content = "# DECISIONS.md\n\n# Day 2 decisions\n"
    marker = "## D2.2 Classification deployment candidate: classical baseline"

    if marker in content:
        start = content.find(marker)
        end_candidates = [
            index
            for index in (
                content.find("\n## D", start + len(marker)),
                content.find("\n# Day ", start + len(marker)),
            )
            if index != -1
        ]
        end = min(end_candidates) if end_candidates else len(content)
        content = f"{content[:start].rstrip()}\n\n{content[end:].lstrip()}".rstrip()

    day2_marker = "# Day 2 decisions"
    if day2_marker not in content:
        content = f"{content.rstrip()}\n\n{day2_marker}\n"

    day2_start = content.find(day2_marker)
    next_day = content.find("\n# Day ", day2_start + len(day2_marker))
    if next_day == -1:
        before = content[: day2_start + len(day2_marker)].rstrip()
        content = f"{before}\n\n{entry}\n"
    else:
        before = content[: day2_start + len(day2_marker)].rstrip()
        after = content[next_day:].lstrip()
        content = f"{before}\n\n{entry}\n\n{after}"
    write_text(content, resolved)


def compare_models(paths: ComparisonPaths | None = None) -> JsonObject:
    if paths is None:
        paths = ComparisonPaths()
    comparison = comparison_payload(
        classical_metrics=read_json(paths.classical_metrics_path),
        distilbert_manifest=read_json(paths.distilbert_manifest_path),
        llm_metrics=read_json(paths.llm_metrics_path),
        paths=paths,
    )
    write_json(comparison, paths.output_path)
    write_text(render_markdown(comparison), paths.report_path)
    upsert_decision_entry(paths.decisions_path, decision_entry(comparison))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classifier baseline metrics.")
    parser.add_argument(
        "--classical-metrics-path",
        type=Path,
        default=DEFAULT_CLASSICAL_METRICS_PATH,
    )
    parser.add_argument(
        "--distilbert-manifest-path",
        type=Path,
        default=DEFAULT_DISTILBERT_MANIFEST_PATH,
    )
    parser.add_argument("--llm-metrics-path", type=Path, default=DEFAULT_LLM_METRICS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--decisions-path", type=Path, default=DEFAULT_DECISIONS_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_models(
        ComparisonPaths(
            classical_metrics_path=args.classical_metrics_path,
            distilbert_manifest_path=args.distilbert_manifest_path,
            llm_metrics_path=args.llm_metrics_path,
            output_path=args.output_path,
            report_path=args.report_path,
            decisions_path=args.decisions_path,
        )
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
