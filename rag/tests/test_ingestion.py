import hashlib
import json
from pathlib import Path

from rag.ingestion import load_corpus, load_queries


def _write_dataset(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "b_doc.txt").write_text("second document text", encoding="utf-8")
    (corpus / "a_doc.txt").write_text("first document text", encoding="utf-8")
    manifest = {
        "docs": {
            "a_doc.txt": hashlib.sha256(b"first document text").hexdigest(),
            "b_doc.txt": hashlib.sha256(b"second document text").hexdigest(),
        },
        "docs_count": 2,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_corpus_is_sorted_and_strips_whitespace(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)
    texts, ids = load_corpus(manifest_path)
    assert ids == ["a_doc", "b_doc"]
    assert texts == ["first document text", "second document text"]


def test_load_corpus_defaults_to_real_data() -> None:
    texts, ids = load_corpus()
    assert len(texts) == len(ids) > 0
    assert "retrieval_augmented_generation" in ids


def test_load_queries_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"id": "q1", "q": "hello", "relevant_ids": ["a_doc"]}\n\n', encoding="utf-8")
    queries = load_queries(path)
    assert queries == [{"id": "q1", "q": "hello", "relevant_ids": ["a_doc"]}]
