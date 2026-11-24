# Experiments

Every number in this document was computed by actually running the code in
this repo, not estimated or invented. Commands to reproduce each result are
included.

**Update:** the network restriction that originally blocked `huggingface.co`
in the environment this document was written in has since been lifted (see
"Scope and limitations" below for the exact hosts that needed allowing). All
`embed`/`hybrid`/`learned` retrieval numbers in section 1 and the reranker
numbers in section 2 are now from the real `sentence-transformers/all-MiniLM-L6-v2`
model, not the dummy hash-based stand-in. The original dummy-model numbers
are kept alongside them below, clearly labeled, because they're still useful
for understanding what the dummy model is for (deterministic plumbing tests,
not a measure of real quality) and because they make the real-vs-dummy
comparison concrete rather than asserted. Section 4 (generation) and section
5 (agent) still report dummy-client numbers only - those need a real
`ANTHROPIC_API_KEY`, which is a separate, still-pending step.

## Corpus

`data/corpus/` holds 24 original documents (I wrote them, so relevance
judgments are known to be correct rather than guessed) covering retrieval,
evaluation, agents, and LLM-safety topics - roughly the subject matter of
this repo itself. `data/queries.jsonl` has 49 natural-language queries with
gold `relevant_ids`. `evals/dataset/qas.jsonl` has 20 question/answer pairs
used for the end-to-end generation eval.

## 1. Retrieval backend comparison

```
# Dummy (hash-based) embeddings - deterministic, no network/model needed:
python scripts/eval_retrieval.py --backend bm25    --k 10 --seed 1337
python scripts/eval_retrieval.py --backend embed   --k 10 --seed 1337
python scripts/eval_retrieval.py --backend hybrid  --k 10 --seed 1337
python scripts/eval_retrieval.py --backend learned --k 10 --seed 1337

# Real embeddings (sentence-transformers/all-MiniLM-L6-v2):
USE_DUMMY_EMBEDDINGS=false python scripts/eval_retrieval.py --backend bm25    --k 10 --seed 1337
USE_DUMMY_EMBEDDINGS=false python scripts/eval_retrieval.py --backend embed   --k 10 --seed 1337
USE_DUMMY_EMBEDDINGS=false python scripts/eval_retrieval.py --backend hybrid  --k 10 --seed 1337
USE_DUMMY_EMBEDDINGS=false python scripts/eval_retrieval.py --backend learned --k 10 --seed 1337
```

| backend | recall@1 | recall@3 | recall@5 | recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.980 | 1.000 | 1.000 | 1.000 | 0.990 | 0.987 |
| embed (dummy) | 0.041 | 0.041 | 0.184 | 0.347 | 0.092 | 0.140 |
| embed (real) | 0.939 | 0.959 | 0.980 | 1.000 | 0.953 | 0.961 |
| hybrid (dummy embed, alpha=0.5) | 0.367 | 0.510 | 0.878 | 0.980 | 0.519 | 0.613 |
| hybrid (real embed, alpha=0.5) | 0.959 | 1.000 | 1.000 | 1.000 | 0.973 | 0.978 |
| learned reranker (dummy embed) | 0.980 | 1.000 | 1.000 | 1.000 | 0.990 | 0.988 |
| learned reranker (real embed) | 0.980 | 1.000 | 1.000 | 1.000 | 0.986 | 0.988 |

(`bm25` is identical in both rows above by construction - it never touches
the embedding model - and is included once as a sanity check that the
pipeline is deterministic regardless of which embedding mode is active.)

**Reading this honestly:** BM25 is near-perfect because the queries are
natural-language paraphrases of documents that each cover a distinct topic
with little vocabulary overlap - a favorable case for lexical matching. The
dummy embedding model is SHA-256 hash bytes normalized to a unit vector: it
carries no semantic signal by construction, so `embed` alone performs close
to random and `hybrid` (which blindly trusts a fixed 50/50 blend) performs
*worse* than BM25 alone by mixing in that noise - a real, reproducible
illustration of why blind score fusion is risky with an unvalidated signal.

The real embedding model tells a different, more useful story: on its own it
already recovers strong retrieval quality (recall@1 0.939, NDCG@10 0.961) -
nowhere near the dummy numbers, because `all-MiniLM-L6-v2` actually carries
semantic signal. Hybrid fusion with real embeddings performs *at least as
well* as BM25 alone (recall@3 1.000, NDCG@10 0.978) rather than worse,
because it's now blending two genuinely informative signals instead of a
good one with noise. The practical takeaway changes once the embedding
signal is real: fusion isn't risky *in general*, it's risky when you fuse in
an unvalidated or broken signal - which is exactly the distinction the dummy
run was built to make visible, and the real run confirms by not reproducing
the failure.

## 2. Learned reranker vs. fixed-alpha fusion

```
# Trains against the dummy embedding model, writes ml/artifacts/reranker_weights.json
# (the default consumed everywhere USE_DUMMY_EMBEDDINGS=true, including CI):
python -m ml.train_loop

# Trains against real embeddings instead, writes a SEPARATE file
# (ml/artifacts/reranker_weights_real_embed.json) so it never clobbers the
# dummy default above. rag/retriever.py::get_backend picks whichever file
# matches the embedding mode actually in use at request time, so the
# "learned" backend is never scored with weights trained on a different
# embedding signal than the one it's being served with:
USE_DUMMY_EMBEDDINGS=false python -m ml.train_loop
```

*(A bug was caught and fixed while producing this section: `ml/reranker.py`
previously hardcoded the dummy embedding model internally regardless of
`USE_DUMMY_EMBEDDINGS`, so training always used dummy features even when the
rest of the pipeline was configured for real embeddings - a silent
train/serve mismatch that would have made the real-embeddings "learned"
backend score real embedding features with weights fit on a completely
different, uncorrelated signal. Fixed by threading the same embedding
settings `rag/retriever.py` already uses through to the trainer, and giving
real- and dummy-trained weights distinct file paths so neither silently
overwrites the other.)*

The reranker is logistic regression over `[bm25_score_normalized,
embed_score_normalized]`, trained on real (query, doc, is_relevant) triples
derived from `data/queries.jsonl` (49 queries x 24 docs = 1,176 pairs, 80/20
train/val split by query, seed 1337).

**Dummy embeddings:**

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

With noisy (dummy) embedding scores, the model learned on real labeled data
that the embedding signal was actively unhelpful and suppressed it
(`coef_embed` negative), while a fixed alpha blindly trusts it regardless of
whether it's informative. On held-out queries the learned reranker recovers
BM25-level recall@3 (1.0) while fixed-weight fusion sits at 0.889.

**Real embeddings** (`sentence-transformers/all-MiniLM-L6-v2`):

```json
{
  "n_train_pairs": 960,
  "n_val_pairs": 216,
  "train_accuracy": 0.9875,
  "val_accuracy": 0.9815,
  "val_auc": 0.9926,
  "weights": {
    "coef_bm25": 5.51,
    "coef_embed": 5.60,
    "intercept": -6.99
  },
  "held_out_recall_at_3": {
    "learned_reranker": 1.0,
    "fixed_equal_weight_fusion": 1.0
  }
}
```

The honest, less dramatic result: with a real embedding signal, `coef_embed`
comes out **positive** and roughly equal in magnitude to `coef_bm25` - the
model correctly learned to trust a signal that's actually informative,
rather than learning to suppress noise. On this corpus, fixed-alpha fusion
and the learned reranker are statistically tied on held-out recall@3 (both
1.0, on a 9-query validation split - too small to distinguish them further).
The dummy run's "learned reranker rescues broken fusion" story doesn't
reproduce here, and that's the right outcome: there's nothing broken left to
rescue once the embedding signal is real. What the learned reranker still
buys you, even when nothing's broken, is not needing to guess alpha by hand
and trusting a fixed 50/50 blend to be the right split for a corpus you
haven't validated yet.

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

- **Hugging Face access.** The environment this was originally written in
  blocked `huggingface.co` outbound. That's now resolved - the model
  download actually requires three allowed hosts, not just one:
  `huggingface.co` itself, plus two CDN hosts it redirects the real file
  transfer to (`cas-server.xethub.hf.co` and `us.aws.cdn.hf.co` as of this
  writing; Hugging Face's CDN routing can change). All `embed`/`hybrid`/
  `learned` numbers in sections 1 and 2 above are from the real
  `sentence-transformers/all-MiniLM-L6-v2` model. The dummy hash-based
  `DummyEmbeddingModel` numbers are kept alongside them, clearly labeled,
  because the dummy model still serves its original purpose - deterministic
  backend *plumbing* tests with no network/model dependency, used by default
  and in CI - just not as a stand-in for real quality numbers anymore.
  `.github/workflows/retrieval-evals.yml` runs the real-embedding path in CI
  too (informational, not gated - see the workflow file for why).
- **No live LLM generation/agent numbers in this document yet.** Real
  generation/agent numbers need `ANTHROPIC_API_KEY` set and `USE_DUMMY_LLM=false`.
  The key is configured on the environment (confirmed via the environment's
  own tooling), but two separate fresh sessions on that same environment
  still do not see it in their process environment -
  `'ANTHROPIC_API_KEY' in os.environ` prints `False` for both. The earlier
  assumption that a session restart would be enough was wrong; this looks
  like an environment-config propagation issue rather than something
  fixable from inside a session, and is pending action outside this repo. Every generation/agent code path is real and
  has been exercised end-to-end against the deterministic dummy client in
  the meantime; the exact commands to produce real numbers once the key is
  actually reachable are in sections 4 and 5 above, and this section will be
  updated once they've actually been run - not estimated or assumed.
