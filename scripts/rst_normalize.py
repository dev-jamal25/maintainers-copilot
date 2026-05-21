"""Tolerant reStructuredText -> plain-text normalization for the Airflow docs corpus (A03).

Airflow docs are ``.rst`` with custom directives, roles, and toctrees. A full docutils parse
is brittle against Airflow's extensions, so this is a deliberately tolerant, line-based
normalizer that preserves what retrieval/generation need and drops navigation noise:

- Section headers (underline, level by first-appearance order, per RST) -> ``#``/``##``/... so the
  chunker can split docs into per-section parents.
- ``.. code-block::`` / literal blocks (``::``) -> fenced code, content preserved.
- Admonitions and unknown directives -> their body text is kept.
- ``.. toctree::`` / ``image`` / ``figure`` / ``raw`` etc. -> dropped (navigation/media).
- Comments, license headers, and ``.. _target:`` anchors -> dropped.
- Inline roles (``:doc:``, ``:ref:`text <t>```), ````literal````, ``*emphasis*`` -> their text.

It also builds the normalized docs corpus (``data/processed/rag_doc_corpus.jsonl``) consumed by
the chunk builder (A06).

Run:
    uv run --project backend python scripts/rst_normalize.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.rag import ChunkSourceType, CorpusRecord  # noqa: E402

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "airflow_docs"
DEFAULT_MANIFEST = DEFAULT_RAW_DIR / "manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "rag_doc_corpus.jsonl"

ADORNMENT_CHARS = set("=-~^\"#*+`':.")
CODE_DIRECTIVES = frozenset({"code-block", "code", "sourcecode", "parsed-literal"})
DROP_DIRECTIVES = frozenset(
    {
        "toctree",
        "image",
        "figure",
        "raw",
        "only",
        "include",
        "literalinclude",
        "contents",
        "highlight",
        "tabs",
        "tab-set",
        "list-table",
        "csv-table",
        "graphviz",
        "mermaid",
        "index",
    }
)

_MARKUP_RE = re.compile(r"^(\s*)\.\.(?:\s+(.*))?\s*$")
_DIRECTIVE_RE = re.compile(r"^([\w+-]+)::(.*)$")
_OPTION_RE = re.compile(r"^:[\w.+-]+:")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_underline(line: str, title: str) -> bool:
    """True when ``line`` is a heading adornment underneath ``title`` (tolerant length)."""
    s = line.strip()
    if len(s) < 3:
        return False
    char = s[0]
    if char not in ADORNMENT_CHARS or any(ch != char for ch in s):
        return False
    if title.endswith("::"):
        return False
    return len(s) >= max(3, len(title) - 3)


def _level_for(order: list[str], char: str) -> int:
    if char not in order:
        order.append(char)
    return order.index(char) + 1


def clean_inline(text: str) -> str:
    """Strip RST inline roles/markup, keeping the human-readable text."""
    # :role:`text <target>`  -> text
    text = re.sub(r":[\w.+-]+:`([^`<]*?)(?:\s*<[^>]*>)?`", r"\1", text)
    # `text <url>`_  (hyperlink) -> text
    text = re.sub(r"`([^`<]*?)\s*<[^>]*>`__?", r"\1", text)
    # ``literal`` -> literal
    text = re.sub(r"``([^`]+)``", r"\1", text)
    # `interpreted` -> interpreted
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # **bold** / *emphasis* -> inner
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # |substitution| -> substitution
    text = re.sub(r"\|([^|]+)\|", r"\1", text)
    return text.rstrip()


def _consume_indented_block(
    lines: list[str], i: int, min_indent: int = 1
) -> tuple[list[str], int]:
    """Collect a dedented indented block starting at ``i``; return (block_lines, next_index)."""
    n = len(lines)
    raw_block: list[str] = []
    while i < n:
        line = lines[i]
        if not line.strip():
            raw_block.append("")
            i += 1
            continue
        if _indent_of(line) >= min_indent:
            raw_block.append(line)
            i += 1
            continue
        break
    while raw_block and not raw_block[0].strip():
        raw_block.pop(0)
    while raw_block and not raw_block[-1].strip():
        raw_block.pop()
    indents = [_indent_of(line) for line in raw_block if line.strip()]
    if not indents:
        return [], i
    cut = min(indents)
    return [line[cut:] if line.strip() else "" for line in raw_block], i


def _skip_markup_block(lines: list[str], i: int, marker_indent: int) -> int:
    """Skip a comment/target line and any more-indented continuation."""
    i += 1
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _indent_of(line) > marker_indent:
            i += 1
            continue
        break
    return i


def _handle_directive(
    name: str, lines: list[str], i: int, indent: int, out: list[str]
) -> int:
    """Process a directive starting at ``i``; append normalized output; return next index."""
    n = len(lines)
    i += 1
    while i < n:
        line = lines[i]
        if (
            line.strip()
            and _OPTION_RE.match(line.strip())
            and _indent_of(line) > indent
        ):
            i += 1
            continue
        break
    while i < n and not lines[i].strip():
        i += 1
    body, i = _consume_indented_block(lines, i, min_indent=indent + 1)
    if name in CODE_DIRECTIVES:
        if body:
            out.append("```")
            out.extend(body)
            out.append("```")
    elif name in DROP_DIRECTIVES:
        pass
    else:  # admonition or unknown directive: keep body as text
        out.extend(clean_inline(line) for line in body)
    return i


def _collapse_blanks(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_rst(text: str) -> str:
    """Normalize a single RST document to plain text with ``#`` headings."""
    lines = text.splitlines()
    out: list[str] = []
    order: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            out.append("")
            i += 1
            continue
        markup = _MARKUP_RE.match(raw)
        if markup:
            indent = len(markup.group(1))
            rest = markup.group(2) or ""
            directive = _DIRECTIVE_RE.match(rest)
            if directive:
                i = _handle_directive(directive.group(1).lower(), lines, i, indent, out)
            else:
                i = _skip_markup_block(lines, i, indent)
            continue
        if i + 1 < n and _is_underline(lines[i + 1], stripped):
            level = _level_for(order, lines[i + 1].strip()[0])
            out.append(f"{'#' * level} {clean_inline(stripped)}")
            i += 2
            continue
        if stripped.endswith("::"):
            lead = clean_inline(stripped[:-2].rstrip())
            if lead:
                out.append(f"{lead}:")
            i += 1
            block, i = _consume_indented_block(lines, i, min_indent=1)
            if block:
                out.append("```")
                out.extend(block)
                out.append("```")
            continue
        out.append(clean_inline(raw.rstrip()))
        i += 1
    return _collapse_blanks("\n".join(out)).strip() + "\n"


def build_doc_corpus_records(raw_dir: Path, manifest_path: Path) -> list[CorpusRecord]:
    """Read raw docs + manifest, normalize each, and build sorted CorpusRecord rows."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest.get("ref", ""))
    records: list[CorpusRecord] = []
    for entry in manifest["docs"]:
        source_id = str(entry["source_id"])
        rst = (raw_dir / f"{source_id}.rst").read_text(encoding="utf-8")
        records.append(
            CorpusRecord(
                source_type=ChunkSourceType.DOCS,
                source_id=source_id,
                text=normalize_rst(rst),
                title=entry.get("title"),
                url=entry.get("url"),
                airflow_area=entry.get("airflow_area"),
                tags=[str(entry["airflow_area"])] if entry.get("airflow_area") else [],
                version=version or None,
            )
        )
    records.sort(key=lambda r: r.source_id)
    return records


def write_corpus_jsonl(records: list[CorpusRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                )
            )
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Airflow RST docs into a corpus JSONL."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    manifest = (
        args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    records = build_doc_corpus_records(raw_dir, manifest)
    write_corpus_jsonl(records, output)
    total_chars = sum(len(record.text) for record in records)
    print(f"normalized docs: {len(records)}")
    print(f"total normalized chars: {total_chars}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
