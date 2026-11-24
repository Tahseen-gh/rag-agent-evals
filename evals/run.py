"""Score the RAG pipeline's generated answers against evals/dataset/qas.jsonl.

Two judges run against each question: a deterministic heuristic judge
(keyword coverage of the gold answer, plus whether the correct source doc
was cited) that needs no API key and always runs, and an optional LLM-judge
pass using a real Anthropic client when --llm-judge is passed and an API key
is reachable (EVAL_ANTHROPIC_API_KEY, falling back to ANTHROPIC_API_KEY -
see fastapi_app/app/config.py:Settings.anthropic_api_key for why the former
takes priority). Earlier versions of this script never called anything - it
wrote "pred": "TBD" for every row without touching the RAG pipeline at all.
This one actually runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from rag.generation import answer_question, get_llm_client
from rag.llm import AnthropicClient, LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "evals" / "dataset" / "qas.jsonl"
RESULTS_PATH = REPO_ROOT / "evals" / "results_run01.jsonl"
SUMMARY_PATH = REPO_ROOT / "evals" / "results_summary.json"

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def heuristic_judge(
    generated_answer: str, gold_answer: str, gold_doc_id: str, cited_doc_ids: list[str]
) -> dict[str, Any]:
    """Deterministic, dependency-free correctness check: does the answer
    mention the gold answer's substantive words, and did it cite the right
    source document? A simplified, offline stand-in for the faithfulness /
    context-recall style checks RAGAS-style evaluation formalizes."""
    gold_words = {w.lower() for w in _WORD_RE.findall(gold_answer) if len(w) > 3}
    gen_lower = generated_answer.lower()
    matched = {w for w in gold_words if w in gen_lower}
    coverage = len(matched) / len(gold_words) if gold_words else 0.0
    cited_correct_doc = gold_doc_id in cited_doc_ids
    correct = coverage >= 0.5 and cited_correct_doc
    return {
        "correct": correct,
        "keyword_coverage": round(coverage, 3),
        "cited_correct_doc": cited_correct_doc,
    }


def llm_judge(
    client: LLMClient, question: str, generated_answer: str, gold_answer: str
) -> dict[str, Any]:
    system = (
        "You are grading whether a candidate answer correctly answers a question, "
        "given a reference answer. Consider the candidate correct if it captures the "
        "same substantive meaning as the reference, even with different wording. "
        "Respond with a first line of exactly 'VERDICT: CORRECT' or "
        "'VERDICT: INCORRECT', then a one-sentence rationale on the next line."
    )
    user = f"Question: {question}\nReference answer: {gold_answer}\nCandidate answer: {generated_answer}"
    response = client.create_message(
        system=system, messages=[{"role": "user", "content": user}], max_tokens=150
    )
    first_line = response.text.strip().splitlines()[0] if response.text.strip() else ""
    correct = "CORRECT" in first_line.upper() and "INCORRECT" not in first_line.upper()
    return {"correct": correct, "rationale": response.text}


def run(
    *,
    backend_name: str = "hybrid",
    k: int = 4,
    use_dummy_embeddings: bool = True,
    use_dummy_llm: bool = True,
    use_llm_judge: bool = False,
    anthropic_api_key: str | None = None,
    llm_model: str = "claude-sonnet-5",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    hybrid_alpha: float = 0.5,
    dataset_path: Path = DATASET,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    gen_client = get_llm_client(use_dummy=use_dummy_llm, api_key=anthropic_api_key, model=llm_model)

    judge_client: LLMClient | None = None
    if use_llm_judge and anthropic_api_key and not use_dummy_llm:
        judge_client = AnthropicClient(api_key=anthropic_api_key, model=llm_model)

    results: list[dict[str, Any]] = []
    for row in dataset:
        result = answer_question(
            row["question"],
            backend_name=backend_name,
            k=k,
            llm_client=gen_client,
            embedding_model=embedding_model,
            hybrid_alpha=hybrid_alpha,
            use_dummy_embeddings=use_dummy_embeddings,
        )
        cited_doc_ids = [c.doc_id for c in result.citations]
        heuristic = heuristic_judge(result.answer, row["answer"], row["doc_id"], cited_doc_ids)
        entry: dict[str, Any] = {
            "id": row.get("id"),
            "question": row["question"],
            "gold_answer": row["answer"],
            "gold_doc_id": row["doc_id"],
            "generated_answer": result.answer,
            "cited_doc_ids": cited_doc_ids,
            "heuristic_judge": heuristic,
        }
        if judge_client is not None:
            entry["llm_judge"] = llm_judge(
                judge_client, row["question"], result.answer, row["answer"]
            )
        results.append(entry)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    n = len(results)
    heuristic_accuracy = (
        sum(1 for r in results if r["heuristic_judge"]["correct"]) / n if n else 0.0
    )
    citation_hit_rate = (
        sum(1 for r in results if r["heuristic_judge"]["cited_correct_doc"]) / n if n else 0.0
    )
    summary: dict[str, Any] = {
        "n": n,
        "backend": backend_name,
        "generator": "dummy_extractive" if use_dummy_llm else llm_model,
        "heuristic_accuracy": round(heuristic_accuracy, 4),
        "citation_hit_rate": round(citation_hit_rate, 4),
    }
    if judge_client is not None:
        llm_accuracy = (
            sum(1 for r in results if r.get("llm_judge", {}).get("correct")) / n if n else 0.0
        )
        summary["llm_judge_accuracy"] = round(llm_accuracy, 4)
        summary["judge_model"] = llm_model
    else:
        summary["llm_judge_accuracy"] = None
        summary["note"] = (
            "LLM judge skipped: pass --llm-judge with EVAL_ANTHROPIC_API_KEY "
            "(or ANTHROPIC_API_KEY) set to enable it."
        )

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results_path} ({n} rows) and {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score RAG-generated answers against evals/dataset/qas.jsonl"
    )
    parser.add_argument(
        "--backend", default="hybrid", choices=["bm25", "embed", "hybrid", "learned"]
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--real-embeddings", dest="use_dummy_embeddings", action="store_false")
    parser.add_argument("--real-llm", dest="use_dummy_llm", action="store_false")
    parser.add_argument("--llm-judge", action="store_true", default=False)
    parser.add_argument("--llm-model", default="claude-sonnet-5")
    parser.set_defaults(use_dummy_embeddings=True, use_dummy_llm=True)
    args = parser.parse_args()

    run(
        backend_name=args.backend,
        k=args.k,
        use_dummy_embeddings=args.use_dummy_embeddings,
        use_dummy_llm=args.use_dummy_llm,
        use_llm_judge=args.llm_judge,
        anthropic_api_key=os.environ.get("EVAL_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY"),
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
