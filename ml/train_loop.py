"""CLI entrypoint: train the learned reranker and report its metrics.

Uses the same USE_DUMMY_EMBEDDINGS/EMBEDDING_MODEL settings as
scripts/eval_retrieval.py, so training matches what get_backend serves.
"""

from __future__ import annotations

import json

from fastapi_app.app.config import settings
from ml.reranker import train_and_evaluate
from rag.backends.learned import DEFAULT_WEIGHTS_PATH, REAL_EMBED_WEIGHTS_PATH


def main() -> dict[str, object]:
    weights_path = (
        DEFAULT_WEIGHTS_PATH if settings.use_dummy_embeddings else REAL_EMBED_WEIGHTS_PATH
    )
    report = train_and_evaluate(
        weights_path=weights_path,
        embedding_model_name=settings.embedding_model,
        use_dummy_embeddings=settings.use_dummy_embeddings,
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
