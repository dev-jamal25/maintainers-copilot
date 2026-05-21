from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CORPUS = _load_script("build_rag_corpus_test_module", "scripts/build_rag_corpus.py")


def test_doc_sources_are_unique_and_bounded() -> None:
    sources = CORPUS.DOC_SOURCES
    assert len(sources) == 30
    CORPUS.assert_unique_source_ids(sources)  # no raise
    ids = [s.source_id for s in sources]
    assert ids == sorted(ids), "DOC_SOURCES must be sorted by source_id for stable output"


def test_all_sources_are_apache_airflow_core_docs() -> None:
    for source in CORPUS.DOC_SOURCES:
        assert source.repo_path.startswith("docs/apache-airflow/")
        assert not source.repo_path.startswith(CORPUS.PROVIDER_DOC_PREFIX)


def test_provider_doc_guard_rejects_provider_paths() -> None:
    bad = (CORPUS.DocSource("p", "providers", "docs/apache-airflow-providers-foo/index.rst", "P"),)
    with pytest.raises(ValueError, match="Provider docs are excluded"):
        CORPUS.assert_no_provider_docs(bad)
    # Curated set must pass the guard.
    CORPUS.assert_no_provider_docs(CORPUS.DOC_SOURCES)


def test_unique_guard_rejects_duplicates() -> None:
    dup = (
        CORPUS.DocSource("x", "faq", "docs/apache-airflow/faq.rst", "X"),
        CORPUS.DocSource("x", "faq", "docs/apache-airflow/best-practices.rst", "X2"),
    )
    with pytest.raises(ValueError, match="Duplicate doc source_ids"):
        CORPUS.assert_unique_source_ids(dup)


def test_raw_url_pins_commit_sha() -> None:
    url = CORPUS.raw_url("docs/apache-airflow/faq.rst")
    assert url == (
        "https://raw.githubusercontent.com/apache/airflow/"
        f"{CORPUS.AIRFLOW_COMMIT_SHA}/docs/apache-airflow/faq.rst"
    )


def test_build_manifest_is_deterministic_and_hashed() -> None:
    source = CORPUS.DocSource("faq", "faq", "docs/apache-airflow/faq.rst", "FAQ")
    content = b"Frequently asked questions\n==========================\n"
    entry = CORPUS.build_manifest_entry(source, content)
    assert entry["source_id"] == "faq"
    assert entry["byte_len"] == len(content)
    assert len(entry["sha256"]) == 64
    assert entry["url"].endswith("/docs/apache-airflow/faq.rst")

    manifest = CORPUS.build_manifest(
        (source,), {"faq": content}, generated_at="2026-05-21T00:00:00+00:00"
    )
    assert manifest["repo"] == "apache/airflow"
    assert manifest["ref"] == CORPUS.AIRFLOW_REF
    assert manifest["commit_sha"] == CORPUS.AIRFLOW_COMMIT_SHA
    assert manifest["provider_docs_excluded"] is True
    assert manifest["doc_count"] == 1
    assert manifest["docs"][0]["sha256"] == entry["sha256"]
