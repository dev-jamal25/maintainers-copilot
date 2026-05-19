from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast, overload

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LABEL_ORDER: tuple[str, ...] = ("bug", "feature", "docs", "question")
LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}
REQUIRED_FIELDS: tuple[str, ...] = (
    "github_id",
    "number",
    "html_url",
    "created_at",
    "label",
    "label_id",
    "text",
)


class ClassificationDataError(ValueError):
    """Raised when classification split data is malformed."""


@dataclass(frozen=True)
class ClassificationExample:
    issue_id: str
    number: int
    html_url: str
    created_at: datetime
    text: str
    label: str
    label_id: int
    split: str
    raw: JsonObject


@dataclass(frozen=True)
class ClassificationDataset(Sequence[ClassificationExample]):
    split: str
    examples: tuple[ClassificationExample, ...]

    def __len__(self) -> int:
        return len(self.examples)

    @overload
    def __getitem__(self, index: int) -> ClassificationExample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ClassificationExample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ClassificationExample | Sequence[ClassificationExample]:
        return self.examples[index]

    def __iter__(self) -> Iterator[ClassificationExample]:
        return iter(self.examples)

    @property
    def texts(self) -> list[str]:
        return [example.text for example in self.examples]

    @property
    def labels(self) -> list[str]:
        return [example.label for example in self.examples]

    @property
    def label_ids(self) -> list[int]:
        return [example.label_id for example in self.examples]


@dataclass(frozen=True)
class ClassificationSplits:
    train: ClassificationDataset
    val: ClassificationDataset
    test: ClassificationDataset


@dataclass(frozen=True)
class ClassificationSplitPaths:
    train: Path = DEFAULT_PROCESSED_DIR / "classification_train.jsonl"
    val: Path = DEFAULT_PROCESSED_DIR / "classification_val.jsonl"
    test: Path = DEFAULT_PROCESSED_DIR / "classification_test.jsonl"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def label_to_id(label: str) -> int:
    try:
        return LABEL_TO_ID[label]
    except KeyError as exc:
        raise ClassificationDataError(f"unknown label {label!r}") from exc


def id_to_label(label_id: int) -> str:
    try:
        return ID_TO_LABEL[label_id]
    except KeyError as exc:
        raise ClassificationDataError(f"unknown label id {label_id!r}") from exc


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed: object = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ClassificationDataError(
                        f"{display_path(path)}:{line_number} contains invalid JSON"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ClassificationDataError(
                        f"{display_path(path)}:{line_number} must be a JSON object"
                    )
                records.append(cast(JsonObject, parsed))
    except OSError as exc:
        raise ClassificationDataError(f"could not read {display_path(path)}: {exc}") from exc
    return records


def require_field(record: JsonObject, field: str, *, path: Path, line_number: int) -> object:
    if field not in record or record[field] is None:
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} missing required field {field!r}"
        )
    return record[field]


def require_string(record: JsonObject, field: str, *, path: Path, line_number: int) -> str:
    value = require_field(record, field, path=path, line_number=line_number)
    if not isinstance(value, str):
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} field {field!r} must be a string"
        )
    return value


def require_int(record: JsonObject, field: str, *, path: Path, line_number: int) -> int:
    value = require_field(record, field, path=path, line_number=line_number)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} field {field!r} must be an int"
        )
    return value


def parse_created_at(value: str, *, path: Path, line_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} field 'created_at' is not parseable"
        ) from exc
    if parsed.tzinfo is None:
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} field 'created_at' must include timezone"
        )
    return parsed


def issue_id(record: JsonObject, *, path: Path, line_number: int) -> str:
    github_id = require_field(record, "github_id", path=path, line_number=line_number)
    return str(github_id)


def validate_required_fields(record: JsonObject, *, path: Path, line_number: int) -> None:
    for field in REQUIRED_FIELDS:
        require_field(record, field, path=path, line_number=line_number)


def parse_example(
    record: JsonObject, *, split: str, path: Path, line_number: int
) -> ClassificationExample:
    validate_required_fields(record, path=path, line_number=line_number)
    label = require_string(record, "label", path=path, line_number=line_number)
    expected_label_id = label_to_id(label)
    label_id = require_int(record, "label_id", path=path, line_number=line_number)
    if label_id != expected_label_id:
        raise ClassificationDataError(
            f"{display_path(path)}:{line_number} label_id {label_id} does not match "
            f"label {label!r} ({expected_label_id})"
        )

    text = normalize_text(require_string(record, "text", path=path, line_number=line_number))
    if not text:
        raise ClassificationDataError(f"{display_path(path)}:{line_number} field 'text' is empty")

    return ClassificationExample(
        issue_id=issue_id(record, path=path, line_number=line_number),
        number=require_int(record, "number", path=path, line_number=line_number),
        html_url=require_string(record, "html_url", path=path, line_number=line_number),
        created_at=parse_created_at(
            require_string(record, "created_at", path=path, line_number=line_number),
            path=path,
            line_number=line_number,
        ),
        text=text,
        label=label,
        label_id=label_id,
        split=split,
        raw=record,
    )


def load_classification_dataset(path: Path, *, split: str) -> ClassificationDataset:
    resolved_path = resolve_repo_path(path)
    examples = tuple(
        parse_example(record, split=split, path=resolved_path, line_number=index + 1)
        for index, record in enumerate(read_jsonl(resolved_path))
    )
    if not examples:
        raise ClassificationDataError(f"{display_path(resolved_path)} does not contain any records")
    return ClassificationDataset(split=split, examples=examples)


def load_classification_splits(
    paths: ClassificationSplitPaths | None = None,
) -> ClassificationSplits:
    if paths is None:
        paths = ClassificationSplitPaths()
    return ClassificationSplits(
        train=load_classification_dataset(paths.train, split="train"),
        val=load_classification_dataset(paths.val, split="val"),
        test=load_classification_dataset(paths.test, split="test"),
    )
