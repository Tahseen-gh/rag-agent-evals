# Evals

`dataset/qas.jsonl` holds real question/answer/source-doc triples used to
score the RAG pipeline end to end. `python -m evals.run --backend hybrid`
runs the pipeline against it and grades each answer with a deterministic
heuristic judge (keyword coverage of the reference answer + whether the
correct source document was cited), or a real LLM-judge with `--llm-judge`
and `ANTHROPIC_API_KEY` set.

Retrieval-only metrics (recall@k, MRR, NDCG@10) are computed separately by
`scripts/eval_retrieval.py` against `data/manifest.json` /
`data/queries.jsonl`, and gated in CI - see the root README's
"Evaluation & CI gates" section.

For real numbers from both, with reproduction commands, see
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
