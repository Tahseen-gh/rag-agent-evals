"""Display-independent logic for the Streamlit UI (ui/app.py).

Deliberately has no Streamlit import: every function here is a plain
function over the same rag/ml modules the API and CLI scripts use, so it
can be unit-tested without a display and reused if a different UI (e.g. a
packaged desktop app) is ever built on top of it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.reranker import train_and_evaluate
from rag.agent import AgentResult, get_agent_llm_client, run_agent
from rag.generation import RagAnswer, answer_question, get_llm_client
from rag.retriever import DOCS_BY_ID, get_backend
from scripts.eval_retrieval import compute_metrics

BACKENDS = ["bm25", "embed", "hybrid", "learned"]
MODES = ["search", "rag_answer", "agent"]
DEFAULT_LLM_MODEL = "claude-sonnet-5"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    text: str


@dataclass(frozen=True)
class ChatOutcome:
    mode: str
    backend: str
    search_results: list[SearchResult] = field(default_factory=list)
    rag_answer: RagAnswer | None = None
    agent_result: AgentResult | None = None


def _search(
    backend_name: str, query: str, k: int, *, use_dummy_embeddings: bool
) -> list[SearchResult]:
    backend = get_backend(
        backend_name,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        hybrid_alpha=0.5,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    return [
        SearchResult(doc_id=doc_id, score=score, text=DOCS_BY_ID[doc_id])
        for doc_id, score in backend.search(query, k=k)
    ]


def run_chat(
    question: str,
    *,
    mode: str,
    backend: str,
    api_key: str | None,
    use_dummy_embeddings: bool = True,
) -> ChatOutcome:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")

    use_dummy_llm = not bool(api_key)

    if mode == "search":
        results = _search(backend, question, 5, use_dummy_embeddings=use_dummy_embeddings)
        return ChatOutcome(mode=mode, backend=backend, search_results=results)

    if mode == "rag_answer":
        client = get_llm_client(use_dummy=use_dummy_llm, api_key=api_key, model=DEFAULT_LLM_MODEL)
        rag_result: RagAnswer = answer_question(
            question,
            backend_name=backend,
            k=4,
            llm_client=client,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            use_dummy_embeddings=use_dummy_embeddings,
        )
        return ChatOutcome(mode=mode, backend=backend, rag_answer=rag_result)

    agent_client = get_agent_llm_client(
        use_dummy=use_dummy_llm, api_key=api_key, model=DEFAULT_LLM_MODEL
    )
    agent_result: AgentResult = run_agent(
        question,
        llm_client=agent_client,
        backend_name=backend,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    return ChatOutcome(mode=mode, backend=backend, agent_result=agent_result)


def run_retrieval_comparison(
    query: str, *, k: int = 3, use_dummy_embeddings: bool = True
) -> dict[str, list[SearchResult]]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return {
        name: _search(name, query, k, use_dummy_embeddings=use_dummy_embeddings)
        for name in BACKENDS
    }


def get_eval_summary(*, k: int = 10, seed: int = 1337) -> dict[str, Any]:
    """Live-recompute retrieval metrics for all four backends and retrain the
    reranker, exactly as documented in docs/EXPERIMENTS.md - same deterministic
    dummy embedding model, same corpus, same seed, so the numbers here match
    that document. Real (non-dummy) embeddings need a Hugging Face model
    download not guaranteed on a free hosted instance, so this always uses
    the dummy model, same as the checked-in reference numbers.
    """
    manifest_path = Path("data/manifest.json")
    retrieval_metrics = {name: compute_metrics(name, k, seed, manifest_path) for name in BACKENDS}

    fd, tmp_path_str = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    weights_path = Path(tmp_path_str)
    try:
        reranker_report = train_and_evaluate(seed=seed, weights_path=weights_path)
    finally:
        weights_path.unlink(missing_ok=True)

    return {"retrieval_metrics": retrieval_metrics, "reranker_report": reranker_report}
