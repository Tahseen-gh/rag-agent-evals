import pytest

from ui.logic import BACKENDS, get_eval_summary, run_chat, run_retrieval_comparison


def test_run_chat_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_chat("   ", mode="search", backend="bm25", api_key=None)


def test_run_chat_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        run_chat("hi", mode="nonsense", backend="bm25", api_key=None)


def test_run_chat_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        run_chat("hi", mode="search", backend="nonsense", api_key=None)


def test_run_chat_search_mode_returns_results() -> None:
    outcome = run_chat("What is BM25?", mode="search", backend="bm25", api_key=None)
    assert outcome.mode == "search"
    assert len(outcome.search_results) == 5
    assert outcome.rag_answer is None
    assert outcome.agent_result is None


def test_run_chat_rag_answer_mode_uses_dummy_client_without_key() -> None:
    outcome = run_chat("What is BM25?", mode="rag_answer", backend="bm25", api_key=None)
    assert outcome.rag_answer is not None
    assert outcome.rag_answer.backend == "bm25"
    assert len(outcome.rag_answer.citations) == 4
    assert "[1]" in outcome.rag_answer.answer


def test_run_chat_agent_mode_uses_dummy_client_without_key() -> None:
    outcome = run_chat("What is BM25?", mode="agent", backend="bm25", api_key=None)
    assert outcome.agent_result is not None
    assert outcome.agent_result.stopped_reason == "end_turn"
    assert outcome.agent_result.steps[0].kind == "tool_call"


def test_run_retrieval_comparison_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_retrieval_comparison("  ")


def test_run_retrieval_comparison_covers_all_backends() -> None:
    results = run_retrieval_comparison("BM25 scoring", k=2)
    assert set(results.keys()) == set(BACKENDS)
    for backend_results in results.values():
        assert len(backend_results) == 2


def test_get_eval_summary_has_all_backends_and_reranker_report() -> None:
    summary = get_eval_summary(k=5, seed=7)
    assert set(summary["retrieval_metrics"].keys()) == set(BACKENDS)
    for metrics in summary["retrieval_metrics"].values():
        assert {"recall@1", "MRR", "NDCG@10"}.issubset(metrics.keys())
    assert "weights" in summary["reranker_report"]
    assert "val_accuracy" in summary["reranker_report"]
