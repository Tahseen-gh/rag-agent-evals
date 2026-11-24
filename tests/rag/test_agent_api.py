from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fastapi_app.app import config
from fastapi_app.app.main import app


def test_agent_ask_endpoint_dummy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "use_dummy_embeddings", True)
    monkeypatch.setattr(config.settings, "use_dummy_llm", True)
    client = TestClient(app)
    r = client.post("/api/v1/agent/ask", json={"question": "What is BM25?", "backend": "bm25"})
    assert r.status_code == 200
    data = r.json()
    assert data["stopped_reason"] == "end_turn"
    assert data["turns_used"] == 2
    assert data["steps"][0]["kind"] == "tool_call"
    assert data["steps"][0]["tool_name"] == "search_corpus"
    assert data["steps"][1]["kind"] == "final_answer"
    assert data["answer"]


def test_agent_ask_endpoint_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "use_dummy_embeddings", True)
    monkeypatch.setattr(config.settings, "use_dummy_llm", True)
    client = TestClient(app)
    r = client.post("/api/v1/agent/ask", json={"question": "anything", "backend": "nonexistent"})
    assert r.status_code == 400
