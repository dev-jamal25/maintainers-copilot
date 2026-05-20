from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from classification.data import (
    LABEL_ORDER,
    REPO_ROOT,
    ClassificationDataset,
    ClassificationExample,
    ClassificationSplitPaths,
    display_path,
    load_classification_dataset,
    resolve_repo_path,
)
from classification.llm_secret_loader import LLMSecretError, SecretLookupResult, load_llm_api_key
from classification.train_transformer import write_json

JsonObject = dict[str, Any]

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "classification" / "llm_baseline"
DEFAULT_PROMPT_PATH = Path(__file__).with_name("llm_baseline_prompt.md")
REPO_ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_LIMIT = 25
TEXT_PREVIEW_LIMIT = 500
ISSUE_TEXT_LIMIT = 12_000

LOGGER = logging.getLogger(__name__)


class LLMResponseError(ValueError):
    """Raised when an LLM response cannot be used as a classification."""


class FatalLLMCallError(RuntimeError):
    """Raised when the provider reports a run-level failure such as auth denial."""


@dataclass(frozen=True)
class ParsedLLMResponse:
    label: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class LLMCallResult:
    parsed: ParsedLLMResponse
    latency_ms: float
    usage: JsonObject


@dataclass(frozen=True)
class SampleSelection:
    examples: tuple[ClassificationExample, ...]
    sampling_mode: str
    limit: int | None
    limit_per_class: int | None
    selected_per_class: dict[str, int]


class IssueClassifier(Protocol):
    model: str

    def classify(self, prompt: str) -> LLMCallResult: ...


def prompt_sha256(prompt_path: Path) -> str:
    digest = hashlib.sha256()
    with prompt_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt(path: Path = DEFAULT_PROMPT_PATH) -> str:
    resolved = resolve_repo_path(path)
    return resolved.read_text(encoding="utf-8").strip()


def text_preview(text: str, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def issue_text_for_prompt(text: str, limit: int = ISSUE_TEXT_LIMIT) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 80]}\n\n[Issue text truncated for classification.]"


def render_prompt(prompt_template: str, example: ClassificationExample) -> str:
    return (
        f"{prompt_template}\n\n"
        "Issue metadata:\n"
        f"- issue_id: {example.issue_id}\n"
        f"- number: {example.number}\n"
        f"- created_at: {example.created_at.isoformat()}\n\n"
        "Issue text:\n"
        f"{issue_text_for_prompt(example.text)}"
    )


def extract_json_object(text: str) -> JsonObject:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMResponseError("response did not contain a JSON object") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError("response JSON could not be parsed") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("response JSON must be an object")
    return cast(JsonObject, parsed)


def parse_llm_json_response(text: str) -> ParsedLLMResponse:
    parsed = extract_json_object(text)
    label = parsed.get("label")
    if not isinstance(label, str) or label not in LABEL_ORDER:
        raise LLMResponseError(f"response label must be one of {', '.join(LABEL_ORDER)}")

    confidence = parsed.get("confidence")
    if isinstance(confidence, int | float) and not isinstance(confidence, bool):
        confidence_value = float(confidence)
    else:
        raise LLMResponseError("response confidence must be a number")
    if not 0.0 <= confidence_value <= 1.0:
        raise LLMResponseError("response confidence must be between 0.0 and 1.0")

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise LLMResponseError("response reason must be a non-empty string")

    return ParsedLLMResponse(
        label=label,
        confidence=confidence_value,
        reason=text_preview(reason, limit=240),
    )


def content_text_from_response(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        value = getattr(block, "text", None)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
    text = "\n".join(parts).strip()
    if not text:
        raise LLMResponseError("provider response did not include text content")
    return text


def token_usage_from_response(response: Any) -> JsonObject:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

    input_count = int(input_tokens or 0)
    output_count = int(output_tokens or 0)
    return {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": input_count + output_count,
    }


def is_fatal_provider_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
    }


class AnthropicIssueClassifier:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, timeout=timeout_seconds)
        self._client = client
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._sleep = sleep
        self.model = model

    def classify(self, prompt: str) -> LLMCallResult:
        last_error: Exception | None = None
        for attempt_index in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.messages.create(
                    max_tokens=self._max_output_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                parsed = parse_llm_json_response(content_text_from_response(response))
                return LLMCallResult(
                    parsed=parsed,
                    latency_ms=latency_ms,
                    usage=token_usage_from_response(response),
                )
            except Exception as exc:
                if is_fatal_provider_error(exc):
                    raise FatalLLMCallError(
                        "LLM provider authentication or permission failed"
                    ) from exc
                last_error = exc
                if attempt_index >= self._max_retries:
                    break
                self._sleep(2.0**attempt_index)

        raise LLMResponseError("LLM call failed after bounded retries") from last_error


def prediction_row(
    *,
    example: ClassificationExample,
    model: str,
    prompt_hash: str,
    result: LLMCallResult | None = None,
    error: str | None = None,
    latency_ms: float | None = None,
) -> JsonObject:
    row: JsonObject = {
        "created_at": example.created_at.isoformat(),
        "html_url": example.html_url,
        "issue_id": example.issue_id,
        "model": model,
        "number": example.number,
        "prompt_sha256": prompt_hash,
        "true_label": example.label,
    }
    if result is not None:
        row.update(
            {
                "confidence": result.parsed.confidence,
                "latency_ms": result.latency_ms,
                "predicted_label": result.parsed.label,
                "reason": result.parsed.reason,
                "usage": result.usage,
            }
        )
    else:
        row.update(
            {
                "confidence": None,
                "error": error or "unknown LLM call failure",
                "latency_ms": latency_ms,
                "predicted_label": None,
                "reason": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        )
    return row


def read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    rows: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(cast(JsonObject, parsed))
    return rows


def append_jsonl(path: Path, row: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def is_successful_prediction(row: Mapping[str, Any]) -> bool:
    return row.get("predicted_label") in LABEL_ORDER and not row.get("error")


def latest_rows_by_issue_id(rows: Iterable[JsonObject]) -> dict[str, JsonObject]:
    latest: dict[str, JsonObject] = {}
    for row in rows:
        issue_id = row.get("issue_id")
        if isinstance(issue_id, str):
            latest[issue_id] = row
    return latest


def completed_prediction_issue_ids(predictions_path: Path) -> set[str]:
    return {
        issue_id
        for issue_id, row in latest_rows_by_issue_id(read_jsonl(predictions_path)).items()
        if is_successful_prediction(row)
    }


def selected_examples(
    dataset: ClassificationDataset, *, limit: int | None
) -> list[ClassificationExample]:
    examples = list(dataset)
    if limit is None:
        return examples
    if limit < 1:
        raise ValueError("--limit must be at least 1 unless --all is used")
    return examples[:limit]


def balanced_selected_examples(
    dataset: ClassificationDataset, *, limit_per_class: int
) -> list[ClassificationExample]:
    if limit_per_class < 1:
        raise ValueError("--limit-per-class must be at least 1")

    selected_by_label: dict[str, list[ClassificationExample]] = {label: [] for label in LABEL_ORDER}
    for example in dataset:
        if len(selected_by_label[example.label]) < limit_per_class:
            selected_by_label[example.label].append(example)

    selected: list[ClassificationExample] = []
    for label in LABEL_ORDER:
        selected.extend(selected_by_label[label])
    return selected


def selected_per_class_counts(examples: Sequence[ClassificationExample]) -> dict[str, int]:
    counts = {label: 0 for label in LABEL_ORDER}
    for example in examples:
        counts[example.label] += 1
    return counts


def sample_dataset(
    dataset: ClassificationDataset,
    *,
    limit: int | None,
    limit_per_class: int | None,
) -> SampleSelection:
    if limit_per_class is not None and limit is not None:
        raise ValueError("--limit-per-class cannot be combined with --limit")

    if limit_per_class is not None:
        examples = balanced_selected_examples(dataset, limit_per_class=limit_per_class)
        return SampleSelection(
            examples=tuple(examples),
            limit=None,
            limit_per_class=limit_per_class,
            sampling_mode="limit_per_class",
            selected_per_class=selected_per_class_counts(examples),
        )

    examples = selected_examples(dataset, limit=limit)
    return SampleSelection(
        examples=tuple(examples),
        limit=limit,
        limit_per_class=None,
        sampling_mode="all" if limit is None else "limit",
        selected_per_class=selected_per_class_counts(examples),
    )


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_usd_per_million: float | None,
    output_price_usd_per_million: float | None,
) -> float | None:
    if input_price_usd_per_million is None or output_price_usd_per_million is None:
        return None
    input_cost = (input_tokens / 1_000_000) * input_price_usd_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_usd_per_million
    return input_cost + output_cost


def compute_metrics(
    rows: Sequence[JsonObject],
    *,
    model: str,
    secret_source: str,
    sampling_mode: str | None = None,
    limit_per_class: int | None = None,
    selected_per_class: Mapping[str, int] | None = None,
    input_price_usd_per_million: float | None = None,
    output_price_usd_per_million: float | None = None,
) -> JsonObject:
    successful_rows = [row for row in rows if is_successful_prediction(row)]
    failed_rows = [row for row in rows if not is_successful_prediction(row)]
    labels = list(LABEL_ORDER)
    truth = [str(row["true_label"]) for row in successful_rows]
    predictions = [str(row["predicted_label"]) for row in successful_rows]
    if successful_rows:
        per_class_values = f1_score(
            truth,
            predictions,
            labels=labels,
            average=None,
            zero_division=0.0,
        )
        matrix = confusion_matrix(truth, predictions, labels=labels).tolist()
        accuracy = float(accuracy_score(truth, predictions))
        macro_f1 = float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0.0)
        )
    else:
        per_class_values = [0.0 for _ in labels]
        matrix = [[0 for _ in labels] for _ in labels]
        accuracy = None
        macro_f1 = None

    latencies = [
        float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), int | float)
    ]
    input_tokens = sum(int(row.get("usage", {}).get("input_tokens", 0) or 0) for row in rows)
    output_tokens = sum(int(row.get("usage", {}).get("output_tokens", 0) or 0) for row in rows)
    metrics: JsonObject = {
        "accuracy": accuracy,
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
        "estimated_cost_usd": estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price_usd_per_million=input_price_usd_per_million,
            output_price_usd_per_million=output_price_usd_per_million,
        ),
        "failed_predictions": len(failed_rows),
        "macro_f1": macro_f1,
        "model": model,
        "per_class_f1": {
            label: float(score) for label, score in zip(labels, per_class_values, strict=True)
        },
        "sampling_mode": sampling_mode,
        "secret_source": secret_source,
        "selected_per_class": dict(selected_per_class or {}),
        "split": "test",
        "successful_predictions": len(successful_rows),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "total_examples_attempted": len(rows),
    }
    if sampling_mode == "limit_per_class":
        metrics["limit_per_class"] = limit_per_class
    return metrics


def rows_for_selected_examples(
    *,
    predictions_path: Path,
    selected_issue_ids: set[str],
) -> list[JsonObject]:
    latest_rows = latest_rows_by_issue_id(read_jsonl(predictions_path))
    return [row for issue_id, row in latest_rows.items() if issue_id in selected_issue_ids]


def run_evaluation(
    *,
    classifier: IssueClassifier,
    dataset: ClassificationDataset,
    output_dir: Path,
    prompt_template: str,
    prompt_hash: str,
    limit: int | None,
    resume: bool,
    secret: SecretLookupResult,
    limit_per_class: int | None = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    input_price_usd_per_million: float | None = None,
    output_price_usd_per_million: float | None = None,
) -> JsonObject:
    resolved_output_dir = resolve_repo_path(output_dir)
    predictions_path = resolved_output_dir / "predictions.jsonl"
    metrics_path = resolved_output_dir / "metrics.json"
    summary_path = resolved_output_dir / "run_summary.json"
    selection = sample_dataset(dataset, limit=limit, limit_per_class=limit_per_class)
    examples = list(selection.examples)
    selected_ids = {example.issue_id for example in examples}

    if not resume:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("", encoding="utf-8")

    completed_ids = completed_prediction_issue_ids(predictions_path) if resume else set()
    skipped = 0
    requested = 0
    for index, example in enumerate(examples, start=1):
        if example.issue_id in completed_ids:
            skipped += 1
            continue

        requested += 1
        LOGGER.info("Classifying issue %s/%s: number=%s", index, len(examples), example.number)
        started = time.perf_counter()
        try:
            result = classifier.classify(render_prompt(prompt_template, example))
            row = prediction_row(
                example=example,
                model=classifier.model,
                prompt_hash=prompt_hash,
                result=result,
            )
        except FatalLLMCallError:
            raise
        except Exception as exc:
            row = prediction_row(
                example=example,
                model=classifier.model,
                prompt_hash=prompt_hash,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        append_jsonl(predictions_path, row)

    rows = rows_for_selected_examples(
        predictions_path=predictions_path,
        selected_issue_ids=selected_ids,
    )
    metrics = compute_metrics(
        rows,
        model=classifier.model,
        secret_source=secret.source,
        sampling_mode=selection.sampling_mode,
        limit_per_class=selection.limit_per_class,
        selected_per_class=selection.selected_per_class,
        input_price_usd_per_million=input_price_usd_per_million,
        output_price_usd_per_million=output_price_usd_per_million,
    )
    summary: JsonObject = {
        "limit": selection.limit,
        "limit_per_class": selection.limit_per_class,
        "metrics_path": display_path(metrics_path),
        "model": classifier.model,
        "output_dir": display_path(resolved_output_dir),
        "predictions_path": display_path(predictions_path),
        "prompt_path": display_path(resolve_repo_path(prompt_path)),
        "prompt_sha256": prompt_hash,
        "requested_this_run": requested,
        "resume": resume,
        "resume_skipped_completed": skipped,
        "sampling_mode": selection.sampling_mode,
        "secret_source": secret.source,
        "secret_source_detail": secret.source_detail,
        "selected_examples": len(examples),
        "selected_per_class": selection.selected_per_class,
        "summary_path": display_path(summary_path),
    }
    write_json(metrics, metrics_path)
    write_json(summary, summary_path)
    return {
        "metrics": metrics,
        "run_summary": summary,
    }


def configured_model_from_env_example(path: Path = REPO_ENV_EXAMPLE_PATH) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("CHATBOT_MODEL="):
            value = stripped.split("=", 1)[1].strip()
            return value or None
    return None


def model_default(env: Mapping[str, str] | None = None) -> str:
    resolved_env = os.environ if env is None else env
    return (
        resolved_env.get("CHATBOT_MODEL", "").strip()
        or configured_model_from_env_example()
        or DEFAULT_MODEL
    )


def validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be at least 1")
    if args.limit_per_class is not None and args.limit is not None:
        parser.error("--limit-per-class cannot be used with --limit")
    if args.limit_per_class is not None and args.all:
        parser.error("--limit-per-class cannot be used with --all")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an offline Anthropic LLM baseline.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full classification test split.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-path", type=Path, default=ClassificationSplitPaths.test)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=model_default())
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--vault-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--input-price-usd-per-million", type=float, default=None)
    parser.add_argument("--output-price-usd-per-million", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    validate_sampling_args(args, parser)
    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=logging.INFO if verbose else logging.WARNING,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    prompt_path = resolve_repo_path(args.prompt_path)
    prompt_template = load_prompt(prompt_path)
    try:
        secret = load_llm_api_key(timeout_seconds=args.vault_timeout_seconds)
    except LLMSecretError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    classifier = AnthropicIssueClassifier(
        api_key=secret.api_key,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    dataset = load_classification_dataset(resolve_repo_path(args.test_path), split="test")
    effective_limit = (
        None
        if args.all or args.limit_per_class is not None
        else (args.limit if args.limit is not None else DEFAULT_LIMIT)
    )
    try:
        result = run_evaluation(
            classifier=classifier,
            dataset=dataset,
            input_price_usd_per_million=args.input_price_usd_per_million,
            limit=effective_limit,
            limit_per_class=args.limit_per_class,
            output_dir=args.output_dir,
            output_price_usd_per_million=args.output_price_usd_per_million,
            prompt_hash=prompt_sha256(prompt_path),
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            resume=args.resume,
            secret=secret,
        )
    except FatalLLMCallError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
