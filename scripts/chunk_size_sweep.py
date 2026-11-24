"""Sweep chunk sizes, measure effect on doc-level retrieval recall.

Chunks each doc at various max_chars, indexes with BM25, checks if the
right doc shows up in the top-k hits. Too small fragments context, too
large dilutes it - see data/corpus/chunking_strategies.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.bm25_index import BM25ChunkIndex
from rag.chunking import chunk_text
from rag.ingestion import load_corpus, load_queries

CHUNK_SIZES = [150, 300, 600, 1200, 2400]
K = 3


def evaluate_chunk_size(max_chars: int) -> dict[str, float]:
    texts, ids = load_corpus()
    queries = load_queries()

    chunks = []
    for doc_id, text in zip(ids, texts, strict=True):
        chunks.extend(chunk_text(doc_id, text, max_chars=max_chars, overlap=max_chars // 6))

    index = BM25ChunkIndex()
    index.build(chunks)

    hits = 0
    for q in queries:
        relevant = set(q["relevant_ids"])
        scored = index.search(str(q["q"]), k=K * 3)
        seen_docs: list[str] = []
        for sc in scored:
            if sc.chunk.doc_id not in seen_docs:
                seen_docs.append(sc.chunk.doc_id)
            if len(seen_docs) >= K:
                break
        if any(doc in relevant for doc in seen_docs):
            hits += 1

    return {
        "max_chars": max_chars,
        "n_chunks": len(chunks),
        "avg_chunks_per_doc": round(len(chunks) / len(ids), 2),
        "recall_at_3_docs": round(hits / len(queries), 4),
    }


def main() -> None:
    results = [evaluate_chunk_size(size) for size in CHUNK_SIZES]
    print(json.dumps(results, indent=2))
    out = Path(__file__).resolve().parent.parent / "evals" / "chunk_size_sweep.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
