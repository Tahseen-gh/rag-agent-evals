from rag.generation import answer_question, get_llm_client
from rag.llm import ExtractiveDummyClient


def test_answer_question_returns_grounded_citations() -> None:
    result = answer_question(
        "What corrections does BM25 make on top of term frequency?",
        backend_name="bm25",
        k=3,
        use_dummy_embeddings=True,
    )
    assert result.backend == "bm25"
    assert len(result.citations) == 3
    assert result.citations[0].doc_id == "sparse_retrieval_bm25"
    assert "[1]" in result.answer


def test_answer_question_invalid_backend_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        answer_question("anything", backend_name="nonexistent")


def test_get_llm_client_defaults_to_dummy_without_key() -> None:
    client = get_llm_client(use_dummy=False, api_key=None, model="claude-sonnet-5")
    assert isinstance(client, ExtractiveDummyClient)


def test_get_llm_client_returns_dummy_when_forced() -> None:
    client = get_llm_client(use_dummy=True, api_key="sk-fake", model="claude-sonnet-5")
    assert isinstance(client, ExtractiveDummyClient)
