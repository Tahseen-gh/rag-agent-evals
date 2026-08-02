# Experiments

Every number in this document was computed by actually running the code in
this repo, not estimated or invented. Commands to reproduce each result are
included. One honest caveat up front: this was written inside a sandboxed
environment whose network policy explicitly blocks `huggingface.co` (see
"Scope and limitations" below), so the real sentence-transformer embedding
model could not be downloaded here. Numbers that depend on it are reported
using the deterministic dummy (hash-based) embedding model instead, and are
labeled as such throughout. Retrieval evaluation code fully supports real
embeddings (`--real-embeddings` / `USE_DUMMY_EMBEDDINGS=false`) - that path
is exercised in `.github/workflows/retrieval-evals.yml` on GitHub Actions,
which is not subject to the same restriction.

## Corpus

`data/corpus/` holds 24 original documents (I wrote them, so relevance
judgments are known to be correct rather than guessed) covering retrieval,
evaluation, agents, and LLM-safety topics - roughly the subject matter of
this repo itself. `data/queries.jsonl` has 49 natural-language queries with
gold `relevant_ids`. `evals/dataset/qas.jsonl` has 20 question/answer pairs
used for the end-to-end generation eval.

## 1. Retrieval backend comparison

```
python scripts/eval_retrieval.py --backend bm25    --k 10 --seed 1337
python scripts/eval_retrieval.py --backend embed   --k 10 --seed 1337
python scripts/eval_retrieval.py --backend hybrid  --k 10 --seed 1337
python scripts/eval_retrieval.py --backend learned --k 10 --seed 1337
```

| backend | recall@1 | recall@3 | recall@5 | recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.980 | 1.000 | 1.000 | 1.000 | 0.990 | 0.987 |
| embed (dummy) | 0.041 | 0.041 | 0.184 | 0.347 | 0.092 | 0.140 |
| hybrid (dummy embed, alpha=0.5) | 0.367 | 0.510 | 0.878 | 0.980 | 0.519 | 0.613 |
| learned reranker (dummy embed) | 0.980 | 1.000 | 1.000 | 1.000 | 0.990 | 0.988 |

**Reading this honestly:** BM25 is near-perfect because the queries are
natural-language paraphrases of documents that each cover a distinct topic
with little vocabulary overlap - a favorable case for lexical matching. The
dummy embedding model is SHA-256 hash bytes normalized to a unit vector: it
carries no semantic signal by construction, so `embed` alone performs close
to random and `hybrid` (which blindly trusts a fixed 50/50 blend) performs
*worse* than BM25 alone by mixing in that noise. This is a real, reproducible
illustration of why blind score fusion is risky: fusing a bad signal with a
good one, without validating the bad signal first, can make things worse,
not better. It is also exactly the failure mode the learned reranker below
was built to catch.

## 2. Learned reranker vs. fixed-alpha fusion

```
python -m ml.train_loop
```

The reranker is logistic regression over `[bm25_score_normalized,
embed_score_normalized]`, trained on real (query, doc, is_relevant) triples
derived from `data/queries.jsonl` (49 queries x 24 docs = 1,176 pairs, 80/20
train/val split by query, seed 1337).

```json
{
  "n_train_pairs": 960,
  "n_val_pairs": 216,
  "train_accuracy": 0.976,
  "val_accuracy": 0.977,
  "val_auc": 0.995,
  "weights": {
    "coef_bm25": 8.52,
    "coef_embed": -1.51,
    "intercept": -4.22
  },
  "held_out_recall_at_3": {
    "learned_reranker": 1.0,
    "fixed_equal_weight_fusion": 0.889
  }
}
```

The interesting result is the sign of `coef_embed`: it's **negative**. Given
noisy (dummy) embedding scores, the model learned on real labeled data that
the embedding signal was actively unhelpful here and suppressed it, while a
fixed alpha blindly trusts it every time regardless of whether it's actually
informative for a given corpus. That's the concrete advantage of a learned
fusion weight over a hand-picked one: it adapts to how good each signal
*actually is*, not how good you assumed it would be. On held-out queries the
learned reranker recovers BM25-level recall@3 (1.0) while fixed-weight
fusion sits at 0.889.

## 3. Chunk size sweep

```
PYTHONPATH=. python scripts/chunk_size_sweep.py
```

| max_chars | chunks produced | avg chunks/doc | doc-level recall@3 |
|---|---|---|---|
| 150 | 400 | 16.7 | 1.000 |
| 300 | 207 | 8.6 | 1.000 |
| 600 | 112 | 4.7 | 1.000 |
| 1200 | 50 | 2.1 | 1.000 |
| 2400 (whole doc) | 24 | 1.0 | 1.000 |

**Honest finding:** doc-level recall@3 didn't move across a 16x range of
chunk sizes on this corpus. That's a real result, not a bug - with 24
topically distinct, well-separated documents, BM25's lexical matching is
robust enough that even a small 150-character fragment still carries the
distinguishing vocabulary needed to find the right document. The tradeoff
chunking strategy actually controls at this scale is operational, not
accuracy: 400 indexed chunks vs. 24, i.e. a 16x difference in index size and
number of BM25/embedding computations per query. I'd expect the accuracy
curve to bend on a larger, noisier, less topically-separated corpus (e.g.
many documents about the same product, where a whole-document embedding
would blur together several unrelated sub-topics) - that's a case this
50-doc-scale corpus doesn't exercise, and worth flagging rather than
overclaiming a tradeoff the data doesn't show.

## 4. End-to-end generation eval

```
python -m evals.run --backend bm25    --k 4
python -m evals.run --backend embed   --k 4
python -m evals.run --backend hybrid  --k 4
python -m evals.run --backend learned --k 4
```

Scored with the deterministic heuristic judge (keyword coverage of the gold
answer >= 50% AND the correct source doc was cited) against 20 held-out
QA pairs, using the dummy extractive generator (no LLM call - see "Scope and
limitations"):

| backend | heuristic accuracy | citation hit rate |
|---|---|---|
| bm25 | 0.70 | 1.00 |
| embed (dummy) | 0.05 | 0.15 |
| hybrid (dummy embed) | 0.50 | 1.00 |
| learned reranker (dummy embed) | 0.70 | 1.00 |

This tracks the pure-retrieval numbers exactly, which is the expected
result for an extractive generator: it can only answer as well as the
context it's given, so generation quality here is a direct readout of
retrieval quality. The citation hit rate for hybrid and learned is 1.00
despite hybrid's heuristic accuracy being lower than bm25/learned - hybrid
still retrieves the *correct document* among its top-4 (citation-eligible)
results often enough, but ranks it lower within the context window, and the
extractive generator's "trust passage [1] first" heuristic then sometimes
extracts from a less relevant passage that happened to rank first (see
`rag/llm.py::ExtractiveDummyClient` for exactly how). A real LLM generator
would likely close some of this gap by actually reading all four passages
rather than only extracting from the first one - that's a concrete,
testable hypothesis for a real (non-dummy) run.

To run this against a real Claude model instead of the dummy extractive
fallback:

```
export ANTHROPIC_API_KEY=sk-...
python -m evals.run --backend hybrid --real-llm --llm-judge
```

`--real-llm` switches the generator itself to the real Anthropic API;
`--llm-judge` additionally uses a real Claude call to grade each answer
against the reference (see `evals/run.py::llm_judge`) instead of relying
solely on the heuristic judge.

## 5. Agent loop (tool-use mechanics)

```
curl -X POST localhost:8000/api/v1/agent/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is BM25?", "backend": "bm25"}'
```

In dummy mode (`USE_DUMMY_LLM=true`, the default), `DummyToolCallClient`
deterministically requests `search_corpus` once, then answers - this
verifies the request/execute/feed-back/continue loop end to end
(`rag/agent.py`, tested in `rag/tests/test_agent.py`) without needing a
model. It cannot demonstrate real agent judgment (when to search again, when
to stop, how to combine multiple tool calls) - that requires
`ANTHROPIC_API_KEY` and `USE_DUMMY_LLM=false`, at which point the identical
loop code runs against the real model.

## Scope and limitations

- **No Hugging Face access in this sandbox.** The environment's outbound
  proxy explicitly denies `huggingface.co` (confirmed via
  `curl "$HTTPS_PROXY/__agentproxy/status"`, which lists it under
  `recentRelayFailures` as a policy denial, not a transient failure). All
  `embed`/`hybrid`/`learned` numbers above therefore use the deterministic
  hash-based `DummyEmbeddingModel`, which carries no real semantic signal by
  design - it exists to make backend *plumbing* deterministically testable,
  not to measure real embedding quality. `.github/workflows/retrieval-evals.yml`
  runs the real-embedding path in CI, which is not behind this restriction;
  the CI gate is deliberately scoped to the BM25 metrics for exactly this
  reason - see the workflow file for the reasoning.
- **No live LLM generation/agent numbers in this document.** No Anthropic
  API key was available in this sandbox (only Claude Code's own internal
  session auth, which isn't a usable general-purpose API key). Every
  generation/agent code path is real and was exercised end-to-end against
  the deterministic dummy client; the exact commands to rerun everything
  above with `ANTHROPIC_API_KEY` set are included in each section.
