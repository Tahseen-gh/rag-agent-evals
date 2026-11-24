from __future__ import annotations

import logging
from typing import Any

from .backends.base import RetrievalBackend
from .backends.bm25 import BM25Backend
from .backends.embed import DummyEmbeddingModel, EmbeddingBackend, EmbeddingModel
from .backends.hybrid import HybridBackend
from .backends.learned import DEFAULT_WEIGHTS_PATH, REAL_EMBED_WEIGHTS_PATH, LearnedRerankBackend
from .ingestion import load_corpus

logger = logging.getLogger(__name__)

_DOC_TEXTS, _DOC_IDS = load_corpus()
DOCS_BY_ID: dict[str, str] = dict(zip(_DOC_IDS, _DOC_TEXTS, strict=True))

_BACKENDS: dict[str, RetrievalBackend] = {}


def _embedding_model(name: str, use_dummy: bool) -> EmbeddingModel:
    if use_dummy:
        return DummyEmbeddingModel()
    from sentence_transformers import SentenceTransformer

    class _STWrapper:
        def __init__(self, name: str):
            self.model = SentenceTransformer(name)

        def encode_texts(self, texts: list[str]) -> Any:
            return self.model.encode(texts, show_progress_bar=False)

    return _STWrapper(name)


def get_backend(
    name: str,
    *,
    embedding_model: str,
    hybrid_alpha: float,
    use_dummy_embeddings: bool,
) -> RetrievalBackend:
    if name in _BACKENDS:
        return _BACKENDS[name]
    backend: RetrievalBackend
    if name == "bm25":
        backend = BM25Backend()
        backend.build(_DOC_TEXTS, _DOC_IDS)
    elif name == "embed":
        model = _embedding_model(embedding_model, use_dummy_embeddings)
        backend = EmbeddingBackend(model)
        backend.build(_DOC_TEXTS, _DOC_IDS)
    elif name == "hybrid":
        bm = BM25Backend()
        bm.build(_DOC_TEXTS, _DOC_IDS)
        model = _embedding_model(embedding_model, use_dummy_embeddings)
        em = EmbeddingBackend(model)
        em.build(_DOC_TEXTS, _DOC_IDS)
        backend = HybridBackend(bm, em, alpha=hybrid_alpha)
    elif name == "learned":
        model = _embedding_model(embedding_model, use_dummy_embeddings)
        # weights must match the embedding signal they were trained on, or it's a
        # silent train/serve mismatch
        weights_path = DEFAULT_WEIGHTS_PATH if use_dummy_embeddings else REAL_EMBED_WEIGHTS_PATH
        if not weights_path.exists():
            logger.warning(
                "No reranker weights at %s (real-embedding reranker not yet trained); "
                "falling back to the dummy-trained weights at %s. Run "
                "`USE_DUMMY_EMBEDDINGS=false python -m ml.train_loop` to fix this.",
                weights_path,
                DEFAULT_WEIGHTS_PATH,
            )
            weights_path = DEFAULT_WEIGHTS_PATH
        backend = LearnedRerankBackend(
            BM25Backend(), EmbeddingBackend(model), weights_path=weights_path
        )
        backend.build(_DOC_TEXTS, _DOC_IDS)
    else:
        raise ValueError(f"unknown backend: {name}")
    _BACKENDS[name] = backend
    return backend
