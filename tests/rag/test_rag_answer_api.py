from __future__ import annotations

from fastapi.testclient import TestClient

from fastapi_app.app import config
from fastapi_app.app.main import app


def test_rag_answer_endpoint_dummy_mode() -> None:
    config.settings.use_dummy_embeddings = True
    config.settings.use_dummy_llm = True
    client = TestClient(app)
    r = client.post(
        "/api/v1/rag/answer",
        json={"question": "What does recall@k measure?", "backend": "bm25", "k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "bm25"
    assert len(data["citations"]) == 3
    assert "[1]" in data["answer"]


def test_rag_answer_endpoint_invalid_backend() -> None:
    config.settings.use_dummy_embeddings = True
    client = TestClient(app)
    r = client.post("/api/v1/rag/answer", json={"question": "anything", "backend": "nonexistent"})
    assert r.status_code == 400


def test_rag_answer_endpoint_rejects_non_positive_k() -> None:
    # k<=0 used to reach faiss's index.search() and crash on an internal
    # assert instead of failing request validation cleanly - caught by fuzzing.
    config.settings.use_dummy_embeddings = True
    client = TestClient(app)
    for k in (0, -1):
        r = client.post(
            "/api/v1/rag/answer", json={"question": "anything", "backend": "bm25", "k": k}
        )
        assert r.status_code == 422


def test_search_endpoint_rejects_non_positive_k() -> None:
    config.settings.use_dummy_embeddings = True
    client = TestClient(app)
    r = client.get("/api/v1/search", params={"q": "anything", "backend": "bm25", "k": 0})
    assert r.status_code == 422
