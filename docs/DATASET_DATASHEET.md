# Datasheet for Datasets — domain-rag-evals

Following the spirit of [Gebru et al., "Datasheets for Datasets"](https://arxiv.org/abs/1803.09010),
scoped to this project's actual corpus and eval set.

## Motivation

The corpus exists to give the retrieval/generation/agent pipeline something
real to search over and be evaluated against, where the right answer is
actually knowable - so retrieval quality can be measured against ground
truth instead of eyeballed. The subject matter (RAG, retrieval, evals,
agents, LLM safety) was deliberately chosen to be the same domain this
project itself is about, so a question like "What is BM25?" has one
unambiguous correct source document.

## Composition

- 24 plain-text (`.txt`) documents in `data/corpus/`, ~700 lines / roughly
  29 lines each on average. English, written at a general technical-reader
  level (assumes familiarity with ML/NLP basics, explains RAG-specific terms
  inline).
- Topics: retrieval (BM25, dense embeddings, hybrid fusion, reranking,
  chunking, vector databases), evaluation (embedding metrics, RAGAS,
  LLM-as-judge), agents (ReAct, tool use, agentic workflows), and LLM safety
  (prompt injection, jailbreaking/red-teaming, constitutional AI, RLHF,
  hallucination mitigation).
- No PII, no copyrighted or scraped material, no sensitive content - every
  document was written from scratch for this project.
- `data/queries.jsonl`: 49 natural-language questions, each with a gold
  `relevant_ids` list (the correct source document(s)) - the labels the
  retrieval and generation evals score against.
- `evals/dataset/qas.jsonl`: 20 question/answer/source-doc triples used for
  the end-to-end generation eval (a held-out subset of the above, with
  reference answers added).

## Collection process

Hand-written for this project, not scraped or generated. Each document's
relevance judgments (which query maps to which doc) were written by the
same person who wrote the document, specifically so the "ground truth"
labels used to compute recall/MRR/NDCG are known-correct by construction
rather than inferred or crowd-sourced. There's a real tradeoff here, stated
plainly rather than hidden: known-correct labels on a small, deliberately
distinct-topic corpus make the eval numbers trustworthy as a measure of
*pipeline correctness*, but this is not a large, messy, real-world corpus -
see `docs/EXPERIMENTS.md` section 3 for what this does and doesn't tell you
(chunk size has no effect on this corpus specifically because each document
is topically self-contained; a corpus with overlapping subject matter would
likely behave differently).

## Preprocessing

- Chunking (`rag/chunking.py`): sentence-boundary-aware splitting with a
  configurable `max_chars` and overlap; `scripts/chunk_size_sweep.py`
  evaluates 150/300/600/1200/2400-char settings.
- No stopword removal or stemming for BM25 (handled internally by
  `rank_bm25`); `clean_text`-style normalization is not applied to this
  corpus (it's clean, hand-written text with no HTML/markup to strip).
- `data/manifest.json` pins a SHA-256 hash of every document's content, and
  tests assert these hashes are unchanged - the corpus is treated as a
  versioned, immutable artifact, not something edited in place.

## Uses

- **Appropriate:** retrieval/RAG/agent evaluation, Q&A over the corpus,
  demonstrating and benchmarking retrieval backends.
- **Inappropriate:** anything requiring a large-scale, real-world, or
  domain-specific (e.g. legal, medical) corpus - this is a small,
  purpose-built eval fixture, not a general-purpose knowledge base. See
  `docs/MODEL_CARD.md`'s "Not for" section for the same caveat at the
  system level.

## Maintenance

Single-maintainer project (see `.github/CODEOWNERS`). The corpus is
versioned via `data/manifest.json`'s content hashes; any content change
requires updating the manifest and re-running the retrieval evals
(`scripts/eval_retrieval.py`) since `data/queries.jsonl`'s relevance
judgments are tied to the current document contents.
