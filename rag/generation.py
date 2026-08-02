"""Retrieve -> select context -> generate a cited answer.

This is the "G" in RAG: rag/retriever.py alone only finds passages, it never
turns them into an answer. answer_question() ties retrieval, chunking (to
keep the prompt focused on the most relevant part of a retrieved document
rather than the whole thing), and generation together, with every claim
expected to carry a citation back to the passage that supports it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_text
from .llm import AnthropicClient, ExtractiveDummyClient, LLMClient
from .retriever import DOCS_BY_ID, get_backend


@dataclass(frozen=True)
class Citation:
    number: int
    doc_id: str
    snippet: str


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    backend: str
    citations: list[Citation]


def get_llm_client(*, use_dummy: bool, api_key: str | None, model: str) -> LLMClient:
    if use_dummy or not api_key:
        return ExtractiveDummyClient()
    return AnthropicClient(api_key=api_key, model=model)


def _select_context(question: str, doc_id: str, text: str, *, max_chars: int = 600) -> str:
    """Pick the passage's most query-relevant chunk instead of sending it whole."""
    chunks = chunk_text(doc_id, text, max_chars=max_chars, overlap=0)
    if not chunks:
        return text[:max_chars]
    q_words = set(question.lower().split())
    best = max(chunks, key=lambda c: len(q_words & set(c.text.lower().split())))
    return best.text


def _build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    numbered = "\n\n".join(
        f"[{i + 1}] {doc_id}: {text}" for i, (doc_id, text) in enumerate(passages)
    )
    return (
        "You are a careful assistant answering questions using only the context "
        "passages below. Every claim in your answer must be supported by one of "
        "the passages. Cite the passage number in square brackets, like [1], "
        "immediately after each claim it supports. If the passages don't contain "
        "the answer, say so instead of guessing.\n\n" + numbered
    )


def answer_question(
    question: str,
    *,
    backend_name: str = "hybrid",
    k: int = 4,
    llm_client: LLMClient | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    hybrid_alpha: float = 0.5,
    use_dummy_embeddings: bool = True,
) -> RagAnswer:
    backend = get_backend(
        backend_name,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    results = backend.search(question, k=k)
    passages = [
        (doc_id, _select_context(question, doc_id, DOCS_BY_ID[doc_id])) for doc_id, _ in results
    ]

    system = _build_prompt(question, passages)
    client = llm_client or ExtractiveDummyClient()
    response = client.create_message(
        system=system, messages=[{"role": "user", "content": question}], max_tokens=512
    )

    citations = [
        Citation(number=i + 1, doc_id=doc_id, snippet=text[:200])
        for i, (doc_id, text) in enumerate(passages)
    ]
    return RagAnswer(
        question=question, answer=response.text, backend=backend_name, citations=citations
    )
