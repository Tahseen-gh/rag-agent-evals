from __future__ import annotations

import builtins
import importlib

import pytest

from rag.backends.embed import DummyEmbeddingModel, EmbeddingBackend


def test_embed_rank_order_deterministic() -> None:
    docs = ["alpha beta", "beta gamma", "delta"]
    ids = ["d1", "d2", "d3"]
    b1 = EmbeddingBackend(DummyEmbeddingModel())
    b1.build(docs, ids, seed=42)
    r1 = b1.search("beta", k=3)
    b2 = EmbeddingBackend(DummyEmbeddingModel())
    b2.build(docs, ids, seed=42)
    r2 = b2.search("beta", k=3)
    assert r1 == r2


def test_embed_fallback_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embed_mod = importlib.reload(importlib.import_module("rag.backends.embed"))
    backend = embed_mod.EmbeddingBackend(embed_mod.DummyEmbeddingModel())
    backend.build(["a", "b"], ["a", "b"])
    assert backend.search("a", k=1)
