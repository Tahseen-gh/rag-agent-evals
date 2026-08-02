import json
from pathlib import Path

from rag.backends.bm25 import BM25Backend
from rag.backends.embed import DummyEmbeddingModel, EmbeddingBackend
from rag.backends.learned import LearnedRerankBackend


def _weights_file(tmp_path: Path, coef_bm25: float, coef_embed: float, intercept: float) -> Path:
    path = tmp_path / "weights.json"
    path.write_text(
        json.dumps({"coef_bm25": coef_bm25, "coef_embed": coef_embed, "intercept": intercept}),
        encoding="utf-8",
    )
    return path


def test_learned_backend_loads_weights_and_scores(tmp_path: Path) -> None:
    weights_path = _weights_file(tmp_path, coef_bm25=1.0, coef_embed=0.0, intercept=0.0)
    docs = ["cats purr softly", "satellites orbit earth", "dogs bark loudly"]
    ids = ["a", "b", "c"]

    bm25 = BM25Backend()
    embed = EmbeddingBackend(DummyEmbeddingModel())
    backend = LearnedRerankBackend(bm25, embed, weights_path=weights_path)
    backend.build(docs, ids)

    results = backend.search("cats", k=2)
    assert len(results) == 2
    assert results[0][0] == "a"
    assert all(0.0 <= score <= 1.0 for _, score in results)


def test_learned_backend_uses_default_committed_weights() -> None:
    docs = ["alpha beta gamma", "delta epsilon zeta"]
    ids = ["d1", "d2"]
    backend = LearnedRerankBackend(BM25Backend(), EmbeddingBackend(DummyEmbeddingModel()))
    backend.build(docs, ids)
    results = backend.search("alpha", k=2)
    assert {doc_id for doc_id, _ in results} == {"d1", "d2"}
