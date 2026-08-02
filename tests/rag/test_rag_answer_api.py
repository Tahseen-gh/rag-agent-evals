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
