from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.json"
DEFAULT_QUERIES = REPO_ROOT / "data" / "queries.jsonl"


def load_corpus(manifest_path: Path = DEFAULT_MANIFEST) -> tuple[list[str], list[str]]:
    """Load the real document corpus described by a manifest.json.

    Returns (texts, ids) in manifest order, sorted by filename for determinism.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs_dir = manifest_path.parent / "corpus"
    texts: list[str] = []
    ids: list[str] = []
    for fname in sorted(manifest["docs"].keys()):
        path = docs_dir / fname
        texts.append(path.read_text(encoding="utf-8").strip())
        ids.append(Path(fname).stem)
    return texts, ids


def load_queries(path: Path = DEFAULT_QUERIES) -> list[dict[str, Any]]:
    """Load newline-delimited JSON queries with 'q' and 'relevant_ids' fields."""
    queries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries
