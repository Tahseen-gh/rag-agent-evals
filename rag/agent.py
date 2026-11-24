"""A minimal tool-calling agent over the retrieval corpus.

Model decides each turn whether to call search_corpus again or answer.
Loop: send, execute any tool call, feed result back, repeat until answer
or turn budget runs out (needed, or a stuck model loops forever).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import AnthropicClient, DummyToolCallClient, LLMClient
from .retriever import DOCS_BY_ID, get_backend

SEARCH_TOOL: dict[str, Any] = {
    "name": "search_corpus",
    "description": (
        "Search the AI-engineering knowledge corpus for passages relevant to a "
        "query. Returns up to k passages, each prefixed with its doc_id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "k": {"type": "integer", "description": "Number of passages to return", "default": 3},
        },
        "required": ["query"],
    },
}

_SYSTEM_PROMPT = (
    "You are a research assistant with access to a search_corpus tool over an "
    "AI-engineering knowledge base. Use the tool as needed to gather evidence, "
    "then give a final answer that cites the doc_ids you relied on in square "
    "brackets, like [doc_id]. If the corpus doesn't contain the answer, say so."
)


@dataclass(frozen=True)
class AgentStep:
    kind: str  # "tool_call" | "final_answer"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class AgentResult:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    turns_used: int = 0
    stopped_reason: str = "end_turn"


def get_agent_llm_client(*, use_dummy: bool, api_key: str | None, model: str) -> LLMClient:
    if use_dummy or not api_key:
        return DummyToolCallClient()
    return AnthropicClient(api_key=api_key, model=model)


def _execute_search_tool(
    query: str,
    k: int,
    *,
    backend_name: str,
    embedding_model: str,
    hybrid_alpha: float,
    use_dummy_embeddings: bool,
) -> str:
    backend = get_backend(
        backend_name,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
        use_dummy_embeddings=use_dummy_embeddings,
    )
    results = backend.search(query, k=max(1, k))
    if not results:
        return "No results found."
    return "\n\n".join(f"[{doc_id}] {DOCS_BY_ID[doc_id][:300]}" for doc_id, _ in results)


def run_agent(
    question: str,
    *,
    llm_client: LLMClient | None = None,
    backend_name: str = "hybrid",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    hybrid_alpha: float = 0.5,
    use_dummy_embeddings: bool = True,
    max_turns: int = 6,
) -> AgentResult:
    client = llm_client or DummyToolCallClient()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[AgentStep] = []

    for turn in range(1, max_turns + 1):
        response = client.create_message(
            system=_SYSTEM_PROMPT, messages=messages, max_tokens=512, tools=[SEARCH_TOOL]
        )

        if response.stop_reason != "tool_use" or not response.tool_calls:
            steps.append(AgentStep(kind="final_answer", text=response.text))
            return AgentResult(
                question=question,
                answer=response.text,
                steps=steps,
                turns_used=turn,
                stopped_reason="end_turn",
            )

        assistant_content: list[dict[str, Any]] = []
        if response.text:
            assistant_content.append({"type": "text", "text": response.text})
        tool_result_blocks: list[dict[str, Any]] = []
        for call in response.tool_calls:
            assistant_content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
            if call.name == "search_corpus":
                output = _execute_search_tool(
                    str(call.input.get("query", question)),
                    int(call.input.get("k", 3)),
                    backend_name=backend_name,
                    embedding_model=embedding_model,
                    hybrid_alpha=hybrid_alpha,
                    use_dummy_embeddings=use_dummy_embeddings,
                )
            else:
                output = f"Unknown tool: {call.name}"
            steps.append(
                AgentStep(
                    kind="tool_call", tool_name=call.name, tool_input=call.input, tool_output=output
                )
            )
            tool_result_blocks.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": output}
            )

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_result_blocks})

    return AgentResult(
        question=question,
        answer="",
        steps=steps,
        turns_used=max_turns,
        stopped_reason="max_turns_exceeded",
    )
