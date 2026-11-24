from __future__ import annotations

from .base import RetrievalBackend, minmax
from .bm25 import BM25Backend
from .embed import EmbeddingBackend


class HybridBackend(RetrievalBackend):
    """Late-fusion of BM25 and embedding scores."""

    def __init__(self, bm25: BM25Backend, embed: EmbeddingBackend, alpha: float = 0.5) -> None:
        self.bm25 = bm25
        self.embed = embed
        self.alpha = alpha

    def build(
        self, docs: list[str], ids: list[str] | None = None, *, seed: int | None = None
    ) -> None:
        self.bm25.build(docs, ids, seed=seed)
        self.embed.build(docs, ids, seed=seed)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        bm = self.bm25.search(query, k)
        em = self.embed.search(query, k)
        ids = sorted({doc_id for doc_id, _ in bm} | {doc_id for doc_id, _ in em})
        bm_scores = {doc_id: score for doc_id, score in bm}
        em_scores = {doc_id: score for doc_id, score in em}
        bm_norm = minmax([bm_scores.get(i, 0.0) for i in ids])
        em_norm = minmax([em_scores.get(i, 0.0) for i in ids])
        fused = [
            (
                ids[i],
                (1 - self.alpha) * bm_norm[i] + self.alpha * em_norm[i],
                bm_scores.get(ids[i], 0.0),
            )
            for i in range(len(ids))
        ]
        fused.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(doc_id, score) for doc_id, score, _ in fused[:k]]
