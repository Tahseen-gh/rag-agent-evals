"""LLM client abstraction shared by rag/generation.py and rag/agent.py.

AnthropicClient hits the real API. ExtractiveDummyClient is the no-key
fallback (default) - just quotes the top passage, no model call, so tests
and CI don't need network access. DummyToolCallClient below scripts a
fixed tool-call-then-answer sequence for testing the agent loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    stop_reason: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Thin wrapper around the real Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return LLMResponse(
            stop_reason=resp.stop_reason or "end_turn",
            text="".join(text_parts),
            tool_calls=tool_calls,
        )


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _keyword_overlap_score(query: str, text: str) -> int:
    q_words = set(w.lower() for w in _WORD_RE.findall(query))
    t_words = set(w.lower() for w in _WORD_RE.findall(text))
    return len(q_words & t_words)


def _best_sentence(query: str, text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text.strip()
    ranked = sorted(sentences, key=lambda s: _keyword_overlap_score(query, s), reverse=True)
    return ranked[0].strip()


class ExtractiveDummyClient:
    """Deterministic, model-free fallback generator.

    Given a RAG-style prompt (system contains context passages formatted as
    "[n] doc_id: text"), it trusts the retriever's ranking - passage [1] is
    assumed most relevant - and answers by extracting the sentence from that
    top passage with the most word overlap with the question, citing it. No
    network, no randomness, no API key needed.
    """

    _PASSAGE_RE = re.compile(r"^\[(\d+)\]\s+(\S+):\s*(.*)$", re.DOTALL)

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        question = str(messages[-1]["content"]) if messages else ""

        passages = self._parse_passages(system)
        if not passages:
            return LLMResponse(
                stop_reason="end_turn", text="I don't have enough context to answer."
            )

        best_num, _doc_id, best_text = passages[0]
        sentence = _best_sentence(question, best_text)
        answer = f"{sentence} [{best_num}]"
        return LLMResponse(stop_reason="end_turn", text=answer)

    def _parse_passages(self, system: str) -> list[tuple[str, str, str]]:
        passages: list[tuple[str, str, str]] = []
        blocks = re.split(r"\n(?=\[\d+\]\s)", system)
        for block in blocks:
            m = self._PASSAGE_RE.match(block.strip())
            if m:
                passages.append((m.group(1), m.group(2), m.group(3)))
        return passages


class DummyToolCallClient:
    """Scripted client for testing the agent's tool-use loop mechanics.

    First call always requests the search_corpus tool with the user's own
    question as the query; once a tool_result is present in the
    conversation, it returns a final answer built from that result. This
    exercises the real request/execute/feed-back/continue loop without a
    real model.
    """

    def __init__(self, tool_name: str = "search_corpus") -> None:
        self.tool_name = tool_name
        self.calls = 0

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(block.get("type") == "tool_result" for block in m["content"])
            for m in messages
        )
        if not has_tool_result:
            question = next((str(m["content"]) for m in messages if m.get("role") == "user"), "")
            return LLMResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="call_1", name=self.tool_name, input={"query": question})],
            )
        return LLMResponse(
            stop_reason="end_turn", text="Based on the search results, here is my answer."
        )
