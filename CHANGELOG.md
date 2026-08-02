# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real 24-document corpus and 49 queries with correct relevance judgments, replacing the synthetic placeholder corpus.
- `rag/generation.py` + `POST /api/v1/rag/answer`: retrieve, select the relevant chunk, generate a cited answer via the Anthropic API (or a deterministic dummy fallback).
- `rag/agent.py` + `POST /api/v1/agent/ask`: a turn-limited tool-calling agent over a `search_corpus` tool.
- `ml/reranker.py`: a logistic-regression "learned" retrieval backend trained on real query-relevance data, replacing the non-functional MLP training stub.
- `evals/run.py` now actually scores generated answers (heuristic + optional LLM judge) instead of writing placeholder predictions.
- `docs/EXPERIMENTS.md`: real, reproducible numbers for every component above.

### Changed
- Project and identifiers renamed from the scaffold-era `openai-ml-phase1`/`phase1`/`phase2` naming to `domain-rag-evals`, reflecting what the project actually is now.
- `retrieval-evals.yml` CI now hard-gates only on the network-free `bm25` backend; `embed`/`hybrid` are reported but not gated, since they depend on a Hugging Face model download.
- `pyproject.toml` `testpaths` now includes the top-level `tests/` directory, which was previously silently excluded from CI.
- `evals-sbom.yml`'s `eval-lite` job (called the now-deleted `rag/eval_metrics.py`) replaced with a `generation-eval` job that runs the real, LLM-judged `evals.run` when `ANTHROPIC_API_KEY` is configured.
- `bandit` (pre-commit and CI) now scans `rag/`, `ml/`, `evals/`, `scripts/` in addition to `fastapi_app/` - previously the bulk of the application logic wasn't covered by the security scan at all.
- Removed unused, orphaned, or empty files left over from earlier project iterations: `rag/eval/` (fixture data for the deleted `eval_metrics.py`), `rag/fixtures.json`, `rag/configs/settings.yaml`, `rag/eval_ragas.py`, `scripts/apply-python-fixes.ps1` (all empty or unreferenced), and a stale `requirements-lock.txt` that had drifted from `requirements.txt` and was never installed from anywhere.

### Fixed
- `data/manifest.json` and `evals/baseline.json` didn't exist despite CI depending on them; the retrieval-evals workflow was broken.
- `docker-compose.yml` had invalid YAML (a healthcheck block mashed onto one line).
- A `mypy` type error in `rag/backends/embed.py`.
- **The Docker image was completely non-functional**: `uvicorn` (the process the image's `CMD` runs) was never listed in `requirements.txt`, the `Dockerfile` never copied `rag/`, `ml/`, or `data/` (only `fastapi_app/`, which imports from all three), and `.env.example` - which the `Dockerfile` copies in as `.env` - set `CORS_ORIGINS` to a bare string where the app requires a JSON array, crashing `pydantic-settings` on startup. All three verified by actually building and booting the app from the exact file set the image now copies, then exercising every endpoint (search across all four backends, `/rag/answer`, `/agent/ask`) against it.
- `.env.example` also had ~9 `APP_*`-prefixed variables left over from an earlier config scheme that don't correspond to any real `Settings` field and were silently ignored - removed, and the file now only contains variables that actually do something.
- `CONTRIBUTING.md` and `SECURITY.md` referenced the deleted `rag/eval_metrics.py` / a renamed CI job.

### Security
-

## [0.1.0] - 2025-08-XX
### Added
- Initial tagged release.

[Unreleased]: #
[0.1.0]: #
