"""CLI entrypoint: train the learned reranker and report its metrics."""

from __future__ import annotations

import json

from ml.reranker import train_and_evaluate


def main() -> dict[str, object]:
    report = train_and_evaluate()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
