from __future__ import annotations

import json
import math
from pathlib import Path

from .base import RetrievalBackend
from .bm25 import BM25Backend
from .embed import EmbeddingBackend

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "reranker_weights.json"
)


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo == 0:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class LearnedRerankBackend(RetrievalBackend):
    """Fuses BM25 + embedding scores using weights learned by ml/reranker.py.

    Same inputs as HybridBackend, but the fusion weight is a logistic
    regression fit on real query-relevance data instead of a fixed alpha.
    """

    def __init__(
        self,
        bm25: BM25Backend,
        embed: EmbeddingBackend,
        *,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
    ) -> None:
        self.bm25 = bm25
        self.embed = embed
        data = json.loads(weights_path.read_text(encoding="utf-8"))
        self.coef_bm25 = float(data["coef_bm25"])
        self.coef_embed = float(data["coef_embed"])
        self.intercept = float(data["intercept"])
        self._ids: list[str] = []

    def build(
        self, docs: list[str], ids: list[str] | None = None, *, seed: int | None = None
    ) -> None:
        self._ids = ids or [str(i) for i in range(len(docs))]
        self.bm25.build(docs, ids, seed=seed)
        self.embed.build(docs, ids, seed=seed)

    def _score(self, bm25_norm: float, embed_norm: float) -> float:
        z = self.coef_bm25 * bm25_norm + self.coef_embed * embed_norm + self.intercept
        return 1.0 / (1.0 + math.exp(-z))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        n = len(self._ids)
        bm_scores = dict(self.bm25.search(query, k=n))
        em_scores = dict(self.embed.search(query, k=n))
        bm_norm = dict(
            zip(self._ids, _minmax([bm_scores.get(i, 0.0) for i in self._ids]), strict=True)
        )
        em_norm = dict(
            zip(self._ids, _minmax([em_scores.get(i, 0.0) for i in self._ids]), strict=True)
        )
        scored = [(i, self._score(bm_norm[i], em_norm[i])) for i in self._ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
