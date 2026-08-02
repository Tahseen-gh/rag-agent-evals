"""Streamlit UI for domain-rag-evals.

Thin on purpose: every button here calls straight into ui/logic.py (which
calls straight into rag/ and ml/, the same code the API and CI use) - no
business logic lives in this file, only rendering.

Run locally: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run ui/app.py` only puts ui/ itself on sys.path, not the repo
# root - add it so `from ui.logic import ...` (and ui.logic's own imports of
# rag/ml) resolve the same way they do under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.logic import ChatOutcome, get_eval_summary, run_chat, run_retrieval_comparison  # noqa: E402

st.set_page_config(page_title="domain-rag-evals", page_icon="🔎", layout="wide")


@st.cache_data(ttl=3600)
def _cached_eval_summary() -> dict[str, Any]:
    # get_eval_summary() retrains the reranker and reruns all four backends
    # over the full query set (~15-20s) - cache so it's not repeated per
    # visitor, only once per hour per deployment. Deterministic (fixed seed),
    # so a cached result is exactly what a fresh computation would return.
    return get_eval_summary()


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        help=(
            "Leave blank to use the deterministic dummy generator - retrieval, "
            "the agent's tool-call loop, and evals all run for real either way. "
            "Paste your own key here to switch generation to real Claude calls "
            "for this browser session only; it is never stored or logged."
        ),
    )
    st.caption(
        "🟢 Live Claude generation"
        if api_key
        else "⚪ Dummy generator (retrieval/agent/evals still real)"
    )
    st.divider()
    st.caption(
        "Source: [domain-rag-evals on GitHub](https://github.com/Tahseen-gh/rag-agent-evals) — "
        "see docs/EXPERIMENTS.md for the full write-up."
    )

page = st.radio(
    "Page", ["Chat", "Retrieval Explorer", "Evals"], horizontal=True, label_visibility="collapsed"
)
st.divider()

if page == "Chat":
    st.subheader("Ask a question over the corpus")
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio(
            "Mode",
            ["search", "rag_answer", "agent"],
            format_func=lambda m: {
                "search": "Search only",
                "rag_answer": "RAG answer (retrieve + generate)",
                "agent": "Agent (may search more than once)",
            }[m],
        )
    with col2:
        backend = st.selectbox("Retrieval backend", ["bm25", "embed", "hybrid", "learned"], index=2)

    question = st.text_input(
        "Question", placeholder="What corrections does BM25 add to term frequency?"
    )
    if st.button("Ask", type="primary") and question:
        outcome: ChatOutcome | None
        with st.spinner("Working..."):
            try:
                outcome = run_chat(question, mode=mode, backend=backend, api_key=api_key or None)
            except ValueError as exc:
                st.error(str(exc))
                outcome = None

        if outcome is not None:
            if outcome.search_results:
                for r in outcome.search_results:
                    with st.expander(f"{r.doc_id}  ·  score {r.score:.3f}"):
                        st.text(r.text)

            if outcome.rag_answer is not None:
                st.markdown(f"**Answer:** {outcome.rag_answer.answer}")
                st.caption("Sources")
                for c in outcome.rag_answer.citations:
                    with st.expander(f"[{c.number}] {c.doc_id}"):
                        st.text(c.snippet)

            if outcome.agent_result is not None:
                st.markdown(f"**Answer:** {outcome.agent_result.answer}")
                st.caption(
                    f"Stopped: {outcome.agent_result.stopped_reason} "
                    f"({outcome.agent_result.turns_used} turn(s))"
                )
                st.caption("Trace")
                for i, step in enumerate(outcome.agent_result.steps, start=1):
                    if step.kind == "tool_call":
                        with st.expander(f"Step {i}: called {step.tool_name}({step.tool_input})"):
                            st.text(step.tool_output)
                    else:
                        st.write(f"Step {i}: final answer")

elif page == "Retrieval Explorer":
    st.subheader("Compare all four backends on one query")
    query = st.text_input("Query", placeholder="hybrid retrieval fusion")
    k = st.slider("Results per backend", 1, 10, 3)
    if st.button("Compare", type="primary") and query:
        try:
            results = run_retrieval_comparison(query, k=k)
        except ValueError as exc:
            st.error(str(exc))
        else:
            cols = st.columns(4)
            for col, name in zip(cols, results.keys(), strict=True):
                with col:
                    st.markdown(f"**{name}**")
                    for r in results[name]:
                        st.markdown(f"`{r.score:.3f}` **{r.doc_id}**")
                        st.text(r.text[:160] + "…")

else:
    st.subheader("Live evals")
    st.caption(
        "Computed live, not screenshotted (cached for an hour after the first visit) - "
        "same corpus, seed, and deterministic dummy embedding model as "
        "docs/EXPERIMENTS.md, so these numbers match that document. Real (non-dummy) "
        "embeddings need a Hugging Face model download not guaranteed on a free "
        "hosted instance."
    )
    with st.spinner("Computing (first visit only - cached after)..."):
        summary = _cached_eval_summary()

    st.markdown("### Retrieval backend comparison")
    metric_names = ["recall@1", "recall@3", "recall@5", "recall@10", "MRR", "NDCG@10"]
    rows = []
    for backend_name, metrics in summary["retrieval_metrics"].items():
        rows.append({"backend": backend_name, **{m: round(metrics[m], 3) for m in metric_names}})
    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("### Learned reranker")
    report = summary["reranker_report"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{report['train_accuracy']:.3f}")
    c2.metric("Val accuracy", f"{report['val_accuracy']:.3f}")
    c3.metric("Val AUC", f"{report['val_auc']:.3f}")
    st.json(report["weights"])
    st.caption(
        "The negative coef_embed weight is the model discovering the dummy embedding "
        "signal is noise and suppressing it - see docs/EXPERIMENTS.md#2 for why."
    )
