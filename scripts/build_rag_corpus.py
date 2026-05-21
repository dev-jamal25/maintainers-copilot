"""Fetch a bounded Apache Airflow documentation corpus for RAG (DECISIONS D3.1).

Acquisition only: this script downloads the raw ``.rst`` source for a fixed, hand-curated
list of doc pages (one per recurring maintainer-support theme in ``rag_holdout.jsonl``) at a
*pinned* Airflow release commit, and writes a provenance manifest with per-file SHA-256
hashes. Normalization happens later in ``scripts/rst_normalize.py`` (A03).

Reproducibility: files are fetched at an immutable commit SHA, so reruns are byte-identical
(verified by the recorded hashes). Provider docs are excluded by default (D3.1); a runtime
guard enforces this.

Run:
    uv run --project backend python scripts/build_rag_corpus.py           # fetch
    uv run --project backend python scripts/build_rag_corpus.py --list    # dry run, no network
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

AIRFLOW_REPO = "apache/airflow"
AIRFLOW_REF = "2.10.3"
# Immutable commit the tag resolves to; pinning the SHA (not the tag) guarantees byte-stable
# reruns even if a tag is ever re-pointed.
AIRFLOW_COMMIT_SHA = "c99887ec11ce3e1a43f2794fcf36d27555140f00"
RAW_BASE = "https://raw.githubusercontent.com"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "airflow_docs"
MANIFEST_NAME = "manifest.json"
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Broad provider docs are excluded (D3.1); guarded at runtime.
PROVIDER_DOC_PREFIX = "docs/apache-airflow-providers"


@dataclass(frozen=True)
class DocSource:
    """One curated documentation source file at the pinned ref."""

    source_id: str
    airflow_area: str
    repo_path: str
    title: str


# Bounded corpus: areas are exactly those enumerated in D3.1; each path was verified to
# resolve at AIRFLOW_COMMIT_SHA. Sorted by source_id for deterministic output.
DOC_SOURCES: tuple[DocSource, ...] = (
    DocSource(
        "authoring_connections",
        "connections",
        "docs/apache-airflow/authoring-and-scheduling/connections.rst",
        "Managing Connections",
    ),
    DocSource(
        "authoring_scheduling_index",
        "scheduling",
        "docs/apache-airflow/authoring-and-scheduling/index.rst",
        "Authoring and Scheduling",
    ),
    DocSource(
        "best_practices",
        "best_practices",
        "docs/apache-airflow/best-practices.rst",
        "Best Practices",
    ),
    DocSource(
        "cli_ref",
        "cli",
        "docs/apache-airflow/cli-and-env-variables-ref.rst",
        "CLI and Environment Variables Reference",
    ),
    DocSource(
        "core_concepts_dag_run",
        "scheduling",
        "docs/apache-airflow/core-concepts/dag-run.rst",
        "DAG Runs",
    ),
    DocSource(
        "core_concepts_dags",
        "core_concepts",
        "docs/apache-airflow/core-concepts/dags.rst",
        "DAGs",
    ),
    DocSource(
        "core_concepts_executor",
        "scheduling",
        "docs/apache-airflow/core-concepts/executor/index.rst",
        "Executor",
    ),
    DocSource(
        "core_concepts_index",
        "core_concepts",
        "docs/apache-airflow/core-concepts/index.rst",
        "Core Concepts",
    ),
    DocSource(
        "core_concepts_operators",
        "operators",
        "docs/apache-airflow/core-concepts/operators.rst",
        "Operators",
    ),
    DocSource(
        "core_concepts_overview",
        "core_concepts",
        "docs/apache-airflow/core-concepts/overview.rst",
        "Architecture Overview",
    ),
    DocSource(
        "core_concepts_params",
        "core_concepts",
        "docs/apache-airflow/core-concepts/params.rst",
        "Params",
    ),
    DocSource(
        "core_concepts_taskflow",
        "core_concepts",
        "docs/apache-airflow/core-concepts/taskflow.rst",
        "TaskFlow",
    ),
    DocSource(
        "core_concepts_tasks",
        "core_concepts",
        "docs/apache-airflow/core-concepts/tasks.rst",
        "Tasks",
    ),
    DocSource(
        "core_concepts_variables",
        "variables",
        "docs/apache-airflow/core-concepts/variables.rst",
        "Variables",
    ),
    DocSource(
        "core_concepts_xcoms",
        "core_concepts",
        "docs/apache-airflow/core-concepts/xcoms.rst",
        "XComs",
    ),
    DocSource(
        "deferring",
        "operators",
        "docs/apache-airflow/authoring-and-scheduling/deferring.rst",
        "Deferrable Operators & Triggers",
    ),
    DocSource(
        "faq",
        "faq",
        "docs/apache-airflow/faq.rst",
        "Frequently Asked Questions",
    ),
    DocSource(
        "howto_connection",
        "connections",
        "docs/apache-airflow/howto/connection.rst",
        "Managing Connections (How-to)",
    ),
    DocSource(
        "howto_index",
        "howto",
        "docs/apache-airflow/howto/index.rst",
        "How-to Guides",
    ),
    DocSource(
        "howto_operator",
        "operators",
        "docs/apache-airflow/howto/operator/index.rst",
        "Using Operators",
    ),
    DocSource(
        "howto_set_config",
        "configuration",
        "docs/apache-airflow/howto/set-config.rst",
        "Setting Configuration Options",
    ),
    DocSource(
        "howto_set_up_database",
        "database",
        "docs/apache-airflow/howto/set-up-database.rst",
        "Set up a Database Backend",
    ),
    DocSource(
        "howto_variable",
        "variables",
        "docs/apache-airflow/howto/variable.rst",
        "Managing Variables (How-to)",
    ),
    DocSource(
        "installation_from_pypi",
        "installation",
        "docs/apache-airflow/installation/installing-from-pypi.rst",
        "Installation from PyPI",
    ),
    DocSource(
        "installation_index",
        "installation",
        "docs/apache-airflow/installation/index.rst",
        "Installation",
    ),
    DocSource(
        "scheduler",
        "scheduling",
        "docs/apache-airflow/administration-and-deployment/scheduler.rst",
        "Scheduler",
    ),
    DocSource(
        "scheduling_cron",
        "scheduling",
        "docs/apache-airflow/authoring-and-scheduling/cron.rst",
        "Cron & Time Intervals",
    ),
    DocSource(
        "security_api",
        "api",
        "docs/apache-airflow/security/api.rst",
        "REST API Security",
    ),
    DocSource(
        "start",
        "installation",
        "docs/apache-airflow/start.rst",
        "Quick Start",
    ),
    DocSource(
        "troubleshooting",
        "troubleshooting",
        "docs/apache-airflow/troubleshooting.rst",
        "Troubleshooting",
    ),
)


def raw_url(repo_path: str, *, commit_sha: str = AIRFLOW_COMMIT_SHA) -> str:
    """Return the immutable raw.githubusercontent URL for a doc path at the pinned commit."""
    return f"{RAW_BASE}/{AIRFLOW_REPO}/{commit_sha}/{repo_path}"


def assert_no_provider_docs(sources: tuple[DocSource, ...]) -> None:
    """Fail loudly if any source is a broad provider doc (excluded by D3.1)."""
    offenders = [
        s.source_id for s in sources if s.repo_path.startswith(PROVIDER_DOC_PREFIX)
    ]
    if offenders:
        raise ValueError(f"Provider docs are excluded by D3.1 but found: {offenders}")


def assert_unique_source_ids(sources: tuple[DocSource, ...]) -> None:
    seen = [s.source_id for s in sources]
    if len(seen) != len(set(seen)):
        dupes = sorted({sid for sid in seen if seen.count(sid) > 1})
        raise ValueError(f"Duplicate doc source_ids: {dupes}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest_entry(source: DocSource, content: bytes) -> dict[str, object]:
    """Build one deterministic manifest entry for a fetched doc."""
    return {
        "source_id": source.source_id,
        "airflow_area": source.airflow_area,
        "repo_path": source.repo_path,
        "title": source.title,
        "url": raw_url(source.repo_path),
        "sha256": sha256_hex(content),
        "byte_len": len(content),
    }


def build_manifest(
    sources: tuple[DocSource, ...],
    contents: dict[str, bytes],
    *,
    generated_at: str,
) -> dict[str, object]:
    """Assemble the corpus manifest; entries sorted by source_id for stable diffs."""
    entries = [
        build_manifest_entry(source, contents[source.source_id])
        for source in sorted(sources, key=lambda s: s.source_id)
        if source.source_id in contents
    ]
    return {
        "repo": AIRFLOW_REPO,
        "ref": AIRFLOW_REF,
        "commit_sha": AIRFLOW_COMMIT_SHA,
        "raw_base": RAW_BASE,
        "generated_at": generated_at,
        "provider_docs_excluded": True,
        "doc_count": len(entries),
        "docs": entries,
    }


async def fetch_doc(client: httpx.AsyncClient, source: DocSource) -> bytes:
    """Fetch a single doc with bounded retry on transient/transport errors."""
    url = raw_url(source.repo_path)
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url)
            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
                return response.content
            if attempt == MAX_RETRIES:
                response.raise_for_status()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == MAX_RETRIES:
                raise
        await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError("unreachable doc fetch retry state")


async def fetch_all(sources: tuple[DocSource, ...]) -> dict[str, bytes]:
    """Fetch every source sequentially (small corpus; keeps GitHub happy)."""
    headers = {"User-Agent": "maintainers-copilot-rag-corpus"}
    timeout = httpx.Timeout(30.0, connect=10.0)
    contents: dict[str, bytes] = {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for source in sorted(sources, key=lambda s: s.source_id):
            contents[source.source_id] = await fetch_doc(client, source)
    return contents


def write_corpus(
    contents: dict[str, bytes],
    manifest: dict[str, object],
    *,
    out_dir: Path,
) -> None:
    """Write each raw .rst and the manifest.json deterministically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for source_id in sorted(contents):
        (out_dir / f"{source_id}.rst").write_bytes(contents[source_id])
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the bounded Airflow RAG docs corpus."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List planned doc sources and exit without fetching.",
    )
    return parser.parse_args()


async def run(out_dir: Path, *, list_only: bool) -> int:
    assert_unique_source_ids(DOC_SOURCES)
    assert_no_provider_docs(DOC_SOURCES)

    if list_only:
        for source in sorted(DOC_SOURCES, key=lambda s: s.source_id):
            print(
                f"{source.source_id:30s} [{source.airflow_area:14s}] {source.repo_path}"
            )
        print(
            f"planned doc sources: {len(DOC_SOURCES)} (ref {AIRFLOW_REF} @ {AIRFLOW_COMMIT_SHA})"
        )
        return 0

    contents = await fetch_all(DOC_SOURCES)
    generated_at = datetime.now(UTC).isoformat()
    manifest = build_manifest(DOC_SOURCES, contents, generated_at=generated_at)
    write_corpus(contents, manifest, out_dir=out_dir)

    total_bytes = sum(len(c) for c in contents.values())
    print(f"fetched docs: {len(contents)} / {len(DOC_SOURCES)}")
    print(f"total bytes: {total_bytes}")
    print(f"ref: {AIRFLOW_REF} @ {AIRFLOW_COMMIT_SHA}")
    print(f"output dir: {out_dir}")
    print(f"manifest: {out_dir / MANIFEST_NAME}")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    return asyncio.run(run(out_dir, list_only=args.list))


if __name__ == "__main__":
    raise SystemExit(main())
