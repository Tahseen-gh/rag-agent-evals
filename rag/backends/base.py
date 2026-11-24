from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrievalBackend(Protocol):
    """Minimal interface for retrieval backends.

    Backends build an index over a corpus and return (doc_id, score) pairs
    for a given query. Implementations must be deterministic when provided
    with the same seed.
    """

    def build(
        self, docs: list[str], ids: list[str] | None = None, *, seed: int | None = None
    ) -> None:
        """Build the index from documents.

        Args:
            docs: List of document texts.
            ids: Optional list of document identifiers. If omitted, sequential
                integer strings will be used.
            seed: Optional deterministic seed.
        """

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs for *query*."""


def minmax(scores: list[float]) -> list[float]:
    """Scale scores to [0, 1]; all-equal input maps to all zeros."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo == 0:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]
