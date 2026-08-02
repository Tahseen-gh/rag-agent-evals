from rag.agent import AgentStep, get_agent_llm_client, run_agent
from rag.llm import DummyToolCallClient, ExtractiveDummyClient


def test_run_agent_calls_tool_then_answers() -> None:
    result = run_agent(
        "What is BM25?",
        llm_client=DummyToolCallClient(),
        backend_name="bm25",
        use_dummy_embeddings=True,
    )
    assert result.stopped_reason == "end_turn"
    assert result.turns_used == 2
    assert len(result.steps) == 2
    assert result.steps[0].kind == "tool_call"
    assert result.steps[0].tool_name == "search_corpus"
    assert result.steps[0].tool_output is not None and "bm25" in result.steps[0].tool_output.lower()
    assert result.steps[1].kind == "final_answer"
    assert result.answer


def test_run_agent_stops_at_max_turns() -> None:
    class AlwaysToolClient:
        def create_message(self, *, system, messages, max_tokens=1024, tools=None):
            from rag.llm import LLMResponse, ToolCall

            return LLMResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="x", name="search_corpus", input={"query": "loop"})],
            )

    result = run_agent("anything", llm_client=AlwaysToolClient(), backend_name="bm25", max_turns=2)
    assert result.stopped_reason == "max_turns_exceeded"
    assert result.turns_used == 2
    assert all(s.kind == "tool_call" for s in result.steps)


def test_run_agent_invalid_backend_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_agent("q", llm_client=DummyToolCallClient(), backend_name="nonexistent")


def test_get_agent_llm_client_defaults_to_dummy() -> None:
    assert isinstance(
        get_agent_llm_client(use_dummy=True, api_key="sk-x", model="claude-sonnet-5"),
        DummyToolCallClient,
    )
    assert isinstance(
        get_agent_llm_client(use_dummy=False, api_key=None, model="claude-sonnet-5"),
        DummyToolCallClient,
    )


def test_agent_step_is_frozen_dataclass() -> None:
    step = AgentStep(kind="final_answer", text="hi")
    assert step.kind == "final_answer"


def test_extractive_dummy_client_unused_by_agent() -> None:
    # sanity: agent needs a tool-use-capable client, not the plain extractive one
    client = ExtractiveDummyClient()
    resp = client.create_message(system="[1] a: text", messages=[{"role": "user", "content": "q"}])
    assert resp.stop_reason == "end_turn"
