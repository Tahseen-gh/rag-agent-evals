import json
from pathlib import Path

from evals.run import heuristic_judge, load_dataset, run


def test_heuristic_judge_correct_when_covered_and_cited() -> None:
    result = heuristic_judge(
        generated_answer="BM25 saturates term frequency and normalizes for document length. [1]",
        gold_answer="it saturates the contribution of very frequent terms and normalizes for document length",
        gold_doc_id="sparse_retrieval_bm25",
        cited_doc_ids=["sparse_retrieval_bm25"],
    )
    assert result["correct"] is True
    assert result["cited_correct_doc"] is True
    assert result["keyword_coverage"] >= 0.5


def test_heuristic_judge_incorrect_when_wrong_doc_cited() -> None:
    result = heuristic_judge(
        generated_answer="BM25 saturates term frequency and normalizes for document length. [1]",
        gold_answer="it saturates the contribution of very frequent terms and normalizes for document length",
        gold_doc_id="sparse_retrieval_bm25",
        cited_doc_ids=["dense_embeddings"],
    )
    assert result["correct"] is False
    assert result["cited_correct_doc"] is False


def test_heuristic_judge_incorrect_when_answer_unrelated() -> None:
    result = heuristic_judge(
        generated_answer="Dogs are great pets. [1]",
        gold_answer="it saturates the contribution of very frequent terms and normalizes for document length",
        gold_doc_id="sparse_retrieval_bm25",
        cited_doc_ids=["sparse_retrieval_bm25"],
    )
    assert result["correct"] is False
    assert result["keyword_coverage"] < 0.5


def test_run_end_to_end_dummy_mode_produces_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qas.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "qa1",
                    "question": "What corrections does BM25 add on top of raw term frequency?",
                    "answer": "it saturates the contribution of very frequent terms and normalizes for document length",
                    "doc_id": "sparse_retrieval_bm25",
                }
            )
            + "\n"
        )

    summary = run(
        backend_name="bm25",
        k=3,
        use_dummy_embeddings=True,
        use_dummy_llm=True,
        dataset_path=dataset_path,
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert summary["n"] == 1
    assert summary["backend"] == "bm25"
    assert summary["llm_judge_accuracy"] is None
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()


def test_load_dataset_reads_real_qas_file() -> None:
    rows = load_dataset()
    assert len(rows) > 0
    assert {"question", "answer", "doc_id"} <= rows[0].keys()
