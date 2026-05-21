from __future__ import annotations

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


RST = _load_script("rst_normalize_test_module", "scripts/rst_normalize.py")


def test_heading_levels_follow_first_appearance_order() -> None:
    rst = "Title\n=====\n\nIntro.\n\nSub\n---\n\nbody\n\nSubSub\n~~~~~~\n\nx\n"
    out = RST.normalize_rst(rst)
    assert "# Title" in out
    assert "## Sub" in out
    assert "### SubSub" in out


def test_faq_style_order_assigns_levels_by_encounter_not_char() -> None:
    rst = "FAQ\n===\n\nSection\n^^^^^^^\n\nq\n\nWhy?\n----\n\na\n"
    out = RST.normalize_rst(rst)
    assert "# FAQ" in out
    assert "## Section" in out  # '^' is the 2nd adornment seen -> level 2
    assert "### Why?" in out  # '-' is the 3rd -> level 3


def test_license_comment_and_target_are_dropped() -> None:
    rst = (
        ".. Licensed to ASF\n   under the terms.\n\n.. _anchor-name:\n\n"
        "Title\n=====\n\nReal content.\n"
    )
    out = RST.normalize_rst(rst)
    assert "Licensed" not in out
    assert "anchor-name" not in out
    assert "# Title" in out
    assert "Real content." in out


def test_code_block_directive_is_fenced_and_options_dropped() -> None:
    rst = (
        "Title\n=====\n\nText:\n\n.. code-block:: python\n   :emphasize-lines: 1\n\n"
        "    import os\n    x = 1\n\nAfter.\n"
    )
    out = RST.normalize_rst(rst)
    assert "```" in out
    assert "import os" in out
    assert "x = 1" in out
    assert ":emphasize-lines:" not in out
    assert "After." in out


def test_toctree_and_image_bodies_are_dropped() -> None:
    rst = (
        "Title\n=====\n\n.. toctree::\n   :maxdepth: 2\n\n   foo\n   bar\n\n"
        ".. image:: /img/x.png\n\nReal content.\n"
    )
    out = RST.normalize_rst(rst)
    assert "foo" not in out
    assert "bar" not in out
    assert "/img/x.png" not in out
    assert "Real content." in out


def test_inline_roles_and_markup_are_unwrapped() -> None:
    rst = (
        "Title\n=====\n\n"
        "See :doc:`tasks` and :ref:`the guide <concepts-x>` plus ``literal`` and *em*.\n"
    )
    out = RST.normalize_rst(rst)
    assert "See tasks and the guide plus literal and em." in out
    assert "`" not in out
    assert ":doc:" not in out


def test_literal_block_marker_becomes_fenced_code() -> None:
    rst = "Title\n=====\n\nExample::\n\n    code line\n    more\n\nDone.\n"
    out = RST.normalize_rst(rst)
    assert "Example:" in out
    assert "Example::" not in out
    assert "code line" in out
    assert "```" in out
    assert "Done." in out


def test_admonition_body_is_kept() -> None:
    rst = "Title\n=====\n\n.. note::\n\n    Remember to set start_date.\n\nNext.\n"
    out = RST.normalize_rst(rst)
    assert "Remember to set start_date." in out
    assert ".. note::" not in out


def test_normalization_is_deterministic() -> None:
    rst = "Title\n=====\n\nSome :ref:`x <y>` text and ``code``.\n\n.. note::\n\n    body\n"
    assert RST.normalize_rst(rst) == RST.normalize_rst(rst)


def test_build_doc_corpus_records_from_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "airflow_docs"
    raw_dir.mkdir()
    (raw_dir / "faq.rst").write_text(
        ".. _faq:\n\nFAQ\n===\n\nWhy?\n----\n\nBecause.\n", encoding="utf-8"
    )
    manifest = {
        "ref": "2.10.3",
        "docs": [
            {
                "source_id": "faq",
                "airflow_area": "faq",
                "title": "FAQ",
                "url": "https://example/faq.rst",
            }
        ],
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    records = RST.build_doc_corpus_records(raw_dir, manifest_path)
    assert len(records) == 1
    record = records[0]
    assert record.source_id == "faq"
    assert record.source_type == "docs"
    assert record.airflow_area == "faq"
    assert record.version == "2.10.3"
    assert record.tags == ["faq"]
    assert "# FAQ" in record.text
    assert "Because." in record.text
