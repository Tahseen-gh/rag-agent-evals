"""Trains a logistic-regression reranker over BM25 + embedding scores,
learning the fusion weight instead of hand-picking alpha. Weights persist
to ml/artifacts/reranker_weights.json, loaded at request time by
rag.backends.learned.LearnedRerankBackend (no sklearn in the serving path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag.backends.base import minmax
from rag.backends.bm25 import BM25Backend
from rag.backends.embed import EmbeddingBackend
from rag.ingestion import load_corpus, load_queries
from rag.retriever import _embedding_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "ml" / "artifacts" / "reranker_weights.json"


@dataclass(frozen=True)
class RerankerWeights:
    coef_bm25: float
    coef_embed: float
    intercept: float

    def score(self, bm25_norm: float, embed_norm: float) -> float:
        z = self.coef_bm25 * bm25_norm + self.coef_embed * embed_norm + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> dict[str, float]:
        return {
            "coef_bm25": self.coef_bm25,
            "coef_embed": self.coef_embed,
            "intercept": self.intercept,
        }


def _per_query_norm_scores(
    ids: list[str], query_text: str, bm25: BM25Backend, embed: EmbeddingBackend
) -> tuple[dict[str, float], dict[str, float]]:
    n = len(ids)
    bm_scores = dict(bm25.search(query_text, k=n))
    em_scores = dict(embed.search(query_text, k=n))
    bm_norm = dict(zip(ids, minmax([bm_scores.get(i, 0.0) for i in ids]), strict=True))
    em_norm = dict(zip(ids, minmax([em_scores.get(i, 0.0) for i in ids]), strict=True))
    return bm_norm, em_norm


def _build_features(
    texts: list[str],
    ids: list[str],
    queries: list[dict[str, Any]],
    *,
    embedding_model_name: str,
    use_dummy_embeddings: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One row per (query, doc) pair: [bm25_norm, embed_norm] -> is_relevant."""
    bm25 = BM25Backend()
    bm25.build(texts, ids)
    embed = EmbeddingBackend(_embedding_model(embedding_model_name, use_dummy_embeddings))
    embed.build(texts, ids)

    rows: list[list[float]] = []
    labels: list[int] = []
    for q in queries:
        relevant = set(q["relevant_ids"])
        bm_norm, em_norm = _per_query_norm_scores(ids, str(q["q"]), bm25, embed)
        for doc_id in ids:
            rows.append([bm_norm[doc_id], em_norm[doc_id]])
            labels.append(1 if doc_id in relevant else 0)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def _recall_at_k(
    texts: list[str],
    ids: list[str],
    queries: list[dict[str, Any]],
    weights: RerankerWeights,
    k: int,
    *,
    embedding_model_name: str,
    use_dummy_embeddings: bool,
) -> float:
    if not queries:
        return float("nan")
    bm25 = BM25Backend()
    bm25.build(texts, ids)
    embed = EmbeddingBackend(_embedding_model(embedding_model_name, use_dummy_embeddings))
    embed.build(texts, ids)
    hits = 0
    for q in queries:
        relevant = set(q["relevant_ids"])
        bm_norm, em_norm = _per_query_norm_scores(ids, str(q["q"]), bm25, embed)
        ranked = sorted(ids, key=lambda i: weights.score(bm_norm[i], em_norm[i]), reverse=True)
        if any(doc in relevant for doc in ranked[:k]):
            hits += 1
    return hits / len(queries)


def train_and_evaluate(
    *,
    seed: int = 1337,
    val_fraction: float = 0.2,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_dummy_embeddings: bool = True,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    texts, ids = load_corpus()
    queries = load_queries()

    rng = np.random.default_rng(seed)
    order = list(range(len(queries)))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_fraction))
    val_idx = set(order[:n_val])
    train_queries = [q for i, q in enumerate(queries) if i not in val_idx]
    val_queries = [q for i, q in enumerate(queries) if i in val_idx]

    x_train, y_train = _build_features(
        texts,
        ids,
        train_queries,
        embedding_model_name=embedding_model_name,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    x_val, y_val = _build_features(
        texts,
        ids,
        val_queries,
        embedding_model_name=embedding_model_name,
        use_dummy_embeddings=use_dummy_embeddings,
    )

    clf = LogisticRegression(class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)

    train_acc = float(accuracy_score(y_train, clf.predict(x_train)))
    val_acc = float(accuracy_score(y_val, clf.predict(x_val)))
    val_probs = clf.predict_proba(x_val)[:, 1]
    val_auc = (
        float(roc_auc_score(y_val, val_probs)) if len(set(y_val.tolist())) > 1 else float("nan")
    )

    weights = RerankerWeights(
        coef_bm25=float(clf.coef_[0][0]),
        coef_embed=float(clf.coef_[0][1]),
        intercept=float(clf.intercept_[0]),
    )

    learned_recall3 = _recall_at_k(
        texts,
        ids,
        val_queries,
        weights,
        k=3,
        embedding_model_name=embedding_model_name,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    fixed_alpha_recall3 = _recall_at_k(
        texts,
        ids,
        val_queries,
        RerankerWeights(coef_bm25=1.0, coef_embed=1.0, intercept=0.0),
        k=3,
        embedding_model_name=embedding_model_name,
        use_dummy_embeddings=use_dummy_embeddings,
    )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(weights.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "embedding_model_name": embedding_model_name,
        "use_dummy_embeddings": use_dummy_embeddings,
        "weights_path": str(weights_path),
        "n_train_pairs": int(len(y_train)),
        "n_val_pairs": int(len(y_val)),
        "n_train_queries": len(train_queries),
        "n_val_queries": len(val_queries),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_auc": val_auc,
        "weights": weights.to_dict(),
        "held_out_recall_at_3": {
            "learned_reranker": learned_recall3,
            "fixed_equal_weight_fusion": fixed_alpha_recall3,
        },
    }
