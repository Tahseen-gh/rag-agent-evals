from rag.llm import DummyToolCallClient, ExtractiveDummyClient, ToolCall


def test_extractive_dummy_client_cites_best_passage() -> None:
    client = ExtractiveDummyClient()
    system = (
        "[1] doc_a: Cats are wonderful companions. They purr softly.\n\n"
        "[2] doc_b: Satellites orbit Earth in low orbit."
    )
    response = client.create_message(
        system=system, messages=[{"role": "user", "content": "Tell me about cats"}]
    )
    assert response.stop_reason == "end_turn"
    assert "[1]" in response.text
    assert "cat" in response.text.lower() or "purr" in response.text.lower()


def test_extractive_dummy_client_handles_no_passages() -> None:
    client = ExtractiveDummyClient()
    response = client.create_message(system="no passages here", messages=[])
    assert "don't have enough context" in response.text


def test_dummy_tool_call_client_first_requests_tool_then_answers() -> None:
    client = DummyToolCallClient()
    first = client.create_message(
        system="s", messages=[{"role": "user", "content": "what is bm25?"}]
    )
    assert first.stop_reason == "tool_use"
    assert first.tool_calls == [
        ToolCall(id="call_1", name="search_corpus", input={"query": "what is bm25?"})
    ]

    second = client.create_message(
        system="s",
        messages=[
            {"role": "user", "content": "what is bm25?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "search_corpus"}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "..."}],
            },
        ],
    )
    assert second.stop_reason == "end_turn"
    assert second.text
