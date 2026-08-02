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

### Fixed
- `data/manifest.json` and `evals/baseline.json` didn't exist despite CI depending on them; the retrieval-evals workflow was broken.
- `docker-compose.yml` had invalid YAML (a healthcheck block mashed onto one line).
- A `mypy` type error in `rag/backends/embed.py`.

### Security
-

## [0.1.0] - 2025-08-XX
### Added
- Initial tagged release.

[Unreleased]: #
[0.1.0]: #
