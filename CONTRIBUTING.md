# Contributing

1. Create a virtualenv and run `make.ps1 setup` (Windows) or `python -m pip install -r requirements.txt`.
2. Pre-commit: `python -m pre_commit install` (hooks run on commit).
3. Run tests: `.\make.ps1 test` (Windows) or `pytest -q` (any).
4. For changes to RAG, update docs/MODEL_CARD.md and run evals: `python scripts/eval_retrieval.py --backend bm25 --seed 1337` (retrieval) and `python -m evals.run --backend hybrid` (generation).
5. Test layout: unit tests for a package live in that package's own `tests/` (e.g. `rag/tests/test_bm25.py`); API/integration tests (anything using `fastapi.testclient.TestClient`) live in the top-level `tests/` tree (e.g. `tests/rag/test_api.py`). Don't duplicate unit-level coverage in both places.
