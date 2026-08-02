# Model Card — domain-rag-evals

**Purpose:** Retrieval-augmented generation and agent backend for answering questions over small corpora, with cited answers.
**Intended use:** Local/dev demo; educational; small internal KBs.
**Not for:** Safety-critical decisions, PHI/PII processing without review.

## System
- Retriever: BM25, dense embeddings, hand-tuned hybrid fusion, and a learned
  (logistic-regression) reranker over sentence/heading-aware chunks.
- Generation: retrieve → select the most relevant chunk → generate a cited
  answer via the Anthropic API (or a deterministic extractive fallback when
  no API key is configured).
- Agent: a turn-limited tool-calling loop that can search the corpus before
  answering.
- Orchestration: FastAPI service + Python libs.
- Data: user-supplied text/markdown; provenance recorded in the datasheet.

## Risks & mitigations
- Hallucinations → keep answers grounded in retrieved chunks; require inline citations; show sources.
- Prompt injection → retrieved content is clearly delimited from instructions in the generation prompt; the agent's tools are limited to read-only corpus search.
- Privacy → don't log raw queries/contexts by default; allow opt-out logging.

## Evaluation
- Retrieval: recall@k, MRR, NDCG@10 on a real, versioned corpus (CI-gated on the BM25 backend).
- Generation: a heuristic judge (keyword coverage + citation correctness) plus an optional LLM-judge pass.
- See `docs/EXPERIMENTS.md` for real, reproducible numbers.

## Limitations
- Small corpus; no sharding/TTL/ACLs.
- The dummy embedding model (used when no network/model access is available) carries no real semantic signal - see `docs/EXPERIMENTS.md` for why this matters.
