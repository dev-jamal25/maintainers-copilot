from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SPLITS = _load_script(
    "build_issue_dataset_splits_test_module",
    "scripts/build_issue_dataset_splits.py",
)
FETCH = _load_script("fetch_github_issues_test_module", "scripts/fetch_github_issues.py")


def test_label_mapping_loads_expected_targets() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")

    assert mapping.repo == "apache/airflow"
    assert mapping.target_labels == {"bug": 0, "feature": 1, "docs": 2, "question": 3}


def test_raw_issue_labels_map_to_expected_target_class() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")

    label, label_id, matched_labels = SPLITS.map_issue_labels(
        [{"name": "kind:feature"}],
        mapping,
    )

    assert label == "feature"
    assert label_id == 1
    assert matched_labels == ["feature"]


def test_pull_request_item_is_skipped() -> None:
    assert FETCH.is_pull_request_item({"number": 123, "pull_request": {"url": "example"}})
    assert not FETCH.is_pull_request_item({"number": 123})


def test_fetch_records_include_source_class_metadata() -> None:
    record = FETCH.extract_issue_record(
        {
            "id": 1,
            "number": 1,
            "title": "Question",
            "body": None,
            "labels": [{"name": "pending-response"}],
            "state": "closed",
            "created_at": "2024-01-01T00:00:00Z",
            "closed_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "html_url": "https://github.com/apache/airflow/issues/1",
        },
        fetched_for_label="question",
        fetched_for_github_label="pending-response",
    )

    assert record["fetched_for_label"] == "question"
    assert record["fetched_for_github_label"] == "pending-response"


def test_fetch_targets_load_from_label_mapping() -> None:
    repo, targets = FETCH.load_label_fetch_targets(
        REPO_ROOT / "ml" / "classification" / "label_mapping.yaml"
    )

    assert repo == "apache/airflow"
    assert {target.target_label for target in targets} == {"bug", "feature", "docs", "question"}
    assert next(target for target in targets if target.target_label == "bug").github_labels == [
        "kind:bug"
    ]


def test_fetch_per_class_target_is_capped_at_500() -> None:
    assert FETCH.resolve_per_class_target(2000, 4) == 500
    assert FETCH.resolve_per_class_target(4000, 4) == 500
    assert FETCH.resolve_per_class_target(40, 4) == 10


def test_sha256_helper_hashes_mapped_output(tmp_path: Path) -> None:
    output = tmp_path / "sample.jsonl"
    output.write_bytes(b'{"label":"bug"}\n')

    assert SPLITS.sha256_file(output) == hashlib.sha256(b'{"label":"bug"}\n').hexdigest()


def test_only_empty_title_and_body_are_dropped() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")
    processed = SPLITS.build_mapped_records(
        [
            {
                "github_id": 1,
                "number": 1,
                "html_url": "https://github.com/apache/airflow/issues/1",
                "created_at": "2024-01-01T00:00:00Z",
                "closed_at": "2024-01-02T00:00:00Z",
                "labels": ["kind:bug"],
                "title": "",
                "body": "",
            }
        ],
        mapping,
    )

    assert processed.records == []
    assert processed.metadata["skipped_by_reason"]["empty_title_and_body"] == 1


def test_short_or_title_only_text_is_kept() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")
    processed = SPLITS.build_mapped_records(
        [
            {
                "github_id": 1,
                "number": 1,
                "html_url": "https://github.com/apache/airflow/issues/1",
                "created_at": "2024-01-01T00:00:00Z",
                "closed_at": "2024-01-02T00:00:00Z",
                "labels": ["kind:bug"],
                "title": "Tiny",
                "body": "",
            }
        ],
        mapping,
    )

    assert len(processed.records) == 1
    assert processed.records[0]["text"] == "Tiny"


def test_fetched_for_label_keeps_pending_response_as_question() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")
    label, label_id, matched_labels, skip_reason = SPLITS.map_issue_to_label(
        {
            "labels": ["kind:bug", "pending-response"],
            "fetched_for_label": "question",
            "fetched_for_github_label": "pending-response",
        },
        mapping,
    )

    assert label == "question"
    assert label_id == 3
    assert matched_labels == ["question"]
    assert skip_reason is None


def test_fetched_for_label_requires_saved_raw_github_label() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")
    label, label_id, matched_labels, skip_reason = SPLITS.map_issue_to_label(
        {
            "labels": ["kind:bug"],
            "fetched_for_label": "question",
            "fetched_for_github_label": "pending-response",
        },
        mapping,
    )

    assert label is None
    assert label_id is None
    assert matched_labels == []
    assert skip_reason == "missing_fetched_github_label"


def test_mapped_records_report_class_shortfalls_without_duplication() -> None:
    mapping = SPLITS.load_label_mapping(REPO_ROOT / "ml" / "classification" / "label_mapping.yaml")
    raw_issues = [
        {
            "github_id": 1,
            "number": 1,
            "html_url": "https://github.com/apache/airflow/issues/1",
            "created_at": "2024-01-01T00:00:00Z",
            "closed_at": "2024-01-02T00:00:00Z",
            "labels": ["kind:bug"],
            "title": "Task instance failure",
            "body": "Scheduler marks a task as failed even after a successful retry.",
        },
        {
            "github_id": 2,
            "number": 2,
            "html_url": "https://github.com/apache/airflow/issues/2",
            "created_at": "2024-01-03T00:00:00Z",
            "closed_at": "2024-01-04T00:00:00Z",
            "labels": ["kind:feature"],
            "title": "Add a scheduler feature",
            "body": "Expose a small configuration hook for custom scheduler behavior.",
        },
    ]

    processed = SPLITS.build_mapped_records(raw_issues, mapping)

    assert len(processed.records) == 2
    assert processed.metadata["mapped_per_class"]["bug"] == 1
    assert processed.metadata["mapped_per_class"]["feature"] == 1
    assert processed.metadata["per_class_shortfalls"]["docs"]["missing"] == 500
    assert processed.metadata["per_class_shortfalls"]["question"]["missing"] == 500


def test_build_dataset_writes_mapped_output_hash_and_no_split_files(tmp_path: Path) -> None:
    raw_path = tmp_path / "github_issues.jsonl"
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    for split_name in ["issues_train.jsonl", "issues_val.jsonl", "issues_test.jsonl"]:
        (output_dir / split_name).write_text("stale\n", encoding="utf-8")

    raw_issue = {
        "github_id": 1,
        "number": 1,
        "html_url": "https://github.com/apache/airflow/issues/1",
        "created_at": "2024-01-01T00:00:00Z",
        "closed_at": "2024-01-02T00:00:00Z",
        "labels": ["pending-response"],
        "fetched_for_label": "question",
        "fetched_for_github_label": "pending-response",
        "title": "How can I configure this operator?",
        "body": "",
    }
    raw_path.write_text(f"{json.dumps(raw_issue)}\n", encoding="utf-8")

    metadata = SPLITS.build_dataset(
        raw_path=raw_path,
        mapping_path=REPO_ROOT / "ml" / "classification" / "label_mapping.yaml",
        output_dir=output_dir,
    )

    mapped_output = output_dir / "issues_mapped.jsonl"
    assert mapped_output.is_file()
    assert metadata["mapped_output_path"] == str(mapped_output)
    assert metadata["mapped_output_sha256"] == SPLITS.sha256_file(mapped_output)
    assert metadata["mapped_issue_count"] == 1
    assert not (output_dir / "issues_train.jsonl").exists()
    assert not (output_dir / "issues_val.jsonl").exists()
    assert not (output_dir / "issues_test.jsonl").exists()
