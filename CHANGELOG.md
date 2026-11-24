# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ui/`: a Streamlit UI (`ui/app.py` + `ui/logic.py`) with a chat page (search/RAG/agent modes with citations), a four-backend retrieval comparison view, and a live evals dashboard computed on demand rather than hardcoded. Defaults to the dummy LLM for safe public hosting; visitors can paste their own Anthropic key in the sidebar to enable real generation for their session.
- Real 24-document corpus and 49 queries with correct relevance judgments, replacing the synthetic placeholder corpus.
- `rag/generation.py` + `POST /api/v1/rag/answer`: retrieve, select the relevant chunk, generate a cited answer via the Anthropic API (or a deterministic dummy fallback).
- `rag/agent.py` + `POST /api/v1/agent/ask`: a turn-limited tool-calling agent over a `search_corpus` tool.
- `ml/reranker.py`: a logistic-regression "learned" retrieval backend trained on real query-relevance data, replacing the non-functional MLP training stub.
- `evals/run.py` now actually scores generated answers (heuristic + optional LLM judge) instead of writing placeholder predictions.
- `docs/EXPERIMENTS.md`: real, reproducible numbers for every component above.
- Real `sentence-transformers/all-MiniLM-L6-v2` embedding numbers for the `embed`/`hybrid`/`learned` retrieval backends and the reranker training run (`docs/EXPERIMENTS.md` sections 1-2), alongside the original dummy-model numbers for comparison.
- Real end-to-end generation numbers (`claude-sonnet-5` generator + LLM judge) for all four retrieval backends (section 4), and a real multi-step agent-loop trace (section 5).
- New `docs/EXPERIMENTS.md` section 6: a real adversarial prompt-injection test (`rag/tests/test_prompt_injection.py`) run against the real model - the agent correctly refused an injected instruction to leak its system prompt and call a nonexistent destructive tool.
- `EVAL_ANTHROPIC_API_KEY` env var, checked before `ANTHROPIC_API_KEY` (`fastapi_app/app/config.py`, `evals/run.py`, the prompt-injection test): some hosting sandboxes reserve the bare `ANTHROPIC_API_KEY` name for their own internal runtime auth and never propagate a user-set value under that name to the process env; this works around that without affecting normal environments (plain `.env`, CI), which have no such collision.
- Real corpus/collection-process content in `docs/DATASET_DATASHEET.md` (was a completely unfilled template).

### Changed
- Project and identifiers renamed from the scaffold-era `openai-ml-phase1`/`phase1`/`phase2` naming to `domain-rag-evals`, reflecting what the project actually is now.
- `retrieval-evals.yml` CI now hard-gates only on the network-free `bm25` backend; `embed`/`hybrid` are reported but not gated, since they depend on a Hugging Face model download.
- `pyproject.toml` `testpaths` now includes the top-level `tests/` directory, which was previously silently excluded from CI.
- `evals-sbom.yml`'s `eval-lite` job (called the now-deleted `rag/eval_metrics.py`) replaced with a `generation-eval` job that runs the real, LLM-judged `evals.run` when `ANTHROPIC_API_KEY` is configured.
- `bandit` (pre-commit and CI) now scans `rag/`, `ml/`, `evals/`, `scripts/`, `ui/` in addition to `fastapi_app/` - previously the bulk of the application logic wasn't covered by the security scan at all.
- `mypy` (pre-commit and CI) now also checks `ui/`.
- `scripts/eval_retrieval.py`: extracted a pure `compute_metrics()` (no file writes, no git subprocess) from `evaluate()`, so the evals UI page reuses the exact same scoring code as the CLI/CI instead of duplicating it.
- Removed unused, orphaned, or empty files left over from earlier project iterations: `rag/eval/` (fixture data for the deleted `eval_metrics.py`), `rag/fixtures.json`, `rag/configs/settings.yaml`, `rag/eval_ragas.py`, `scripts/apply-python-fixes.ps1` (all empty or unreferenced), and a stale `requirements-lock.txt` that had drifted from `requirements.txt` and was never installed from anywhere.
- `evals/CHANGELOG.md` renamed to `evals/BASELINE_LOG.md`: it was never a changelog, just `scripts/update_baseline.py`'s append-only log of baseline updates, and shared a confusing name with this file.
- Misplaced test file `tests/rag/test_backends.py` (unit-testing `EmbeddingBackend`/`HybridBackend` directly rather than via the API, duplicating the scope of `rag/tests/`) split into `rag/tests/test_embed.py` / `test_hybrid.py`, matching the existing per-backend naming convention; the unit-vs-integration test-location rule documented in `CONTRIBUTING.md`.
- `actions/checkout` bumped v4 -> v5 across all 5 workflow files.
- `make.ps1`'s `setup` now installs `requirements-dev.txt` too, matching the Makefile's `install` (was silently missing `pytest-cov` and friends on Windows).
- `fastapi_app/app/main.py`: replaced the deprecated `@app.on_event("startup")` hook with a `lifespan` context manager (same startup behavior - `configure_logging` + `init_otel` - just on the API FastAPI itself expects going forward).
- `_minmax()` was defined identically in three places (`rag/backends/hybrid.py`, `rag/backends/learned.py`, `ml/reranker.py`); consolidated into one `minmax()` in `rag/backends/base.py`, imported by all three.
- `requirements.txt`: dropped `pytest` (a dev/test tool, not a runtime dependency - already declared correctly in `requirements-dev.txt`) and `ragas` (listed but never imported anywhere in the codebase). Bumped `fastapi` to a pinned `0.128.0` and `prometheus-fastapi-instrumentator` to `>=7.0.0,<8.0.0` to resolve a `starlette` CVE without pulling in `starlette>=1.0`, which `prometheus-fastapi-instrumentator>=8.0` requires but the pinned `schemathesis` (and the `api-fuzz.yml` workflow built on it) does not yet support - see `SECURITY.md` for the residual, intentionally-deferred part of this.
- Added `rag/backends/__init__.py` and `fastapi_app/app/api/__init__.py` - both packages worked via Python's implicit namespace-package support, but every other package in the repo has an explicit `__init__.py` (even empty ones), so this was just an inconsistency.

### Fixed
- `rag/chunking.py`: `chunk_text()` joined sentence spans without their separating whitespace, silently gluing chunk-internal sentence boundaries together (e.g. "sentence.It" instead of "sentence. It"). Found via manual UI testing, not by any prior automated test; added a regression test.
- `data/manifest.json` and `evals/baseline.json` didn't exist despite CI depending on them; the retrieval-evals workflow was broken.
- `docker-compose.yml` had invalid YAML (a healthcheck block mashed onto one line).
- A `mypy` type error in `rag/backends/embed.py`.
- **The Docker image was completely non-functional**: `uvicorn` (the process the image's `CMD` runs) was never listed in `requirements.txt`, the `Dockerfile` never copied `rag/`, `ml/`, or `data/` (only `fastapi_app/`, which imports from all three), and `.env.example` - which the `Dockerfile` copies in as `.env` - set `CORS_ORIGINS` to a bare string where the app requires a JSON array, crashing `pydantic-settings` on startup. All three verified by actually building and booting the app from the exact file set the image now copies, then exercising every endpoint (search across all four backends, `/rag/answer`, `/agent/ask`) against it.
- `.env.example` also had ~9 `APP_*`-prefixed variables left over from an earlier config scheme that don't correspond to any real `Settings` field and were silently ignored - removed, and the file now only contains variables that actually do something.
- `CONTRIBUTING.md` and `SECURITY.md` referenced the deleted `rag/eval_metrics.py` / a renamed CI job.
- `ml/reranker.py` hardcoded the dummy embedding model internally regardless of `USE_DUMMY_EMBEDDINGS`, so the "learned" backend could silently score real embedding features with weights trained on a completely different (dummy) signal - a train/serve mismatch. Real- and dummy-trained weights now persist to separate files (`rag/retriever.py` loads whichever matches the active embedding mode).
- `ml/tests/test_train_loop.py::test_main_writes_default_artifact` called `main()` with no path override, silently overwriting the real checked-in `ml/artifacts/reranker_weights.json` on every test run - now uses a `tmp_path`.
- The prompt-injection test's own first assertion was over-strict: it banned any mention of the attempted tool's name in the agent's answer, which fired on the model naming-and-refusing the attack (correct behavior), not on compliance. Corrected to check for actual compliance language instead.
- `SECURITY.md` had a literal unfilled `email SECURITY_CONTACT (replace)` placeholder - replaced with GitHub's private security advisory flow.
- Closed 6 long-stale Dependabot PRs (#5-#10, opened 2025-08-29): all but #5 only touched `requirements-lock.txt`, deleted months ago in this project's rewrite, or bumped pins no longer used in their exact-pinned form; all had diverged too far from `main` to merge (confirmed via merge-conflict errors on both the merge and branch-update APIs) regardless.
- `fastapi_app/app/api/v1.py` declared its own `GET /health` returning `{"status": "ok"}`, but `fastapi_app/app/main.py` registers its own `/api/v1/health` *before* mounting that router, so the router's handler was permanently shadowed and unreachable in the real app (verified by hitting the live app - `main.py`'s handler always won). It was only "passing" via `fastapi_app/tests/test_api_router.py`, which built the router into its own standalone `FastAPI()` instance rather than testing the real app. Removed the dead route and its isolation-only test.
- `docs/EXPERIMENTS.md` section 3 referred to "this 50-doc-scale corpus," but the corpus is 24 documents everywhere else in the repo (manifest, datasheet, README) - a leftover inconsistency in a document whose whole premise is "every number here is measured, not asserted." Corrected to 24.
- Deprecated `datetime.datetime.utcnow()` replaced with `datetime.datetime.now(datetime.UTC)` in `scripts/update_baseline.py` and `tests/evals/test_eval_scripts.py`.
- `tests/rag/test_api.py`, `test_agent_api.py`, `test_rag_answer_api.py` mutated the shared `fastapi_app.app.config.settings` singleton directly with no teardown; switched to `monkeypatch.setattr`, which pytest restores automatically after each test.

### Security
- `pip-audit` flagged a CVE on the pinned `pytest==8.4.1` in `requirements.txt`; resolved by removing it from runtime deps entirely (see Changed).
- `pip-audit` also flagged several CVEs on the `starlette` version pulled in by the previously-pinned `fastapi==0.116.1`; bumping `fastapi` to `0.128.0` resolves one of them (the rest need a `starlette>=1.0` migration that isn't safe to do as part of this pass - see `SECURITY.md`'s new "Known pip-audit findings" section for the full reasoning and what's blocking it).

## [0.1.0] - 2025-08-XX
### Added
- Initial tagged release.

[Unreleased]: #
[0.1.0]: #
