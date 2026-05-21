"""Maintainer's Copilot RAG library (ML-heavy: chunking, embeddings, retrieval, rerank).

The ``ml`` project is ``package = false`` and runs with the repo's ``ml/`` directory on
``sys.path`` (so ``rag`` imports as a top-level package). Importing this package also puts the
repository root on ``sys.path`` so the canonical contract at ``backend/app/domain/rag.py`` is
importable from every ml entrypoint and test without duplicating schemas.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
