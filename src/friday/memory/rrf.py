"""Reciprocal Rank Fusion (RRF) and hybrid (vector + keyword) recall.

This module is dependency-injected and self-contained: it imports nothing from
``friday.config`` or ``friday.app`` and takes every collaborator as a function
parameter. That keeps it composable and trivially testable.

**Reciprocal Rank Fusion.** Given several independent rankings of the *same*
universe of items (each ranking a list of item ids, best first), RRF assigns
each item a score of ``sum(1 / (k + rank))`` across the rankings it appears in,
where ``rank`` is the item's 0-based position in that ranking and ``k`` is a
smoothing constant (default 60, the value from the original Cormack et al.
paper). Items are then returned best-score first. RRF needs no score
calibration between the input rankers — it only uses ranks — which is exactly
why it is the standard way to fuse a vector (semantic) ranking with a keyword
(lexical) ranking.

**HybridRecall.** A thin orchestrator that calls an injected ``vector_search``
and an injected ``keyword_search`` (each ``(text, k) -> list[str]`` of ids,
best first), fuses their outputs with :func:`reciprocal_rank_fusion`, and
returns the top ``k`` fused ids.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one, by Reciprocal Rank Fusion.

    Args:
        rankings: A list of rankings. Each ranking is a list of item ids ordered
            best-first. The same id may appear in several rankings; within a
            single ranking ids are expected to be unique and only the first
            occurrence of a duplicate is scored.
        k: RRF smoothing constant. Must be >= 1. Larger ``k`` flattens the
            contribution of top ranks (reduces the influence of any single
            ranker's first place); the literature default is 60.

    Returns:
        A list of ``(id, score)`` pairs sorted by score descending. Ties are
        broken deterministically by id (ascending) so the output is stable.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking):
            if item_id in seen:
                # Only the best (first) position of a duplicate id counts.
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    # Sort by score descending, then by id ascending for deterministic ties.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class SearchFn(Protocol):
    """A ranked-search callable: ``(text, k) -> list[str]`` of ids, best first."""

    def __call__(self, text: str, k: int) -> list[str]: ...


class HybridRecall:
    """Fuse a vector (semantic) and a keyword (lexical) ranker via RRF.

    Both searchers are injected; this class imports neither and knows nothing
    about how ids are produced. Each searcher returns a list of ids ordered
    best-first.

    Args:
        vector_search: Semantic ranker, ``(text, k) -> list[str]``.
        keyword_search: Lexical ranker, ``(text, k) -> list[str]``.
        k: RRF smoothing constant applied during fusion (default 60).
    """

    def __init__(
        self,
        vector_search: SearchFn,
        keyword_search: SearchFn,
        k: int = DEFAULT_K,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._vector_search = vector_search
        self._keyword_search = keyword_search
        self._k = k

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        """Return the top ``k`` ids fused from both rankers, best score first.

        Args:
            text: The query text passed verbatim to both injected searchers.
            k: Number of fused results to return *and* the per-ranker fetch
                width passed to each searcher. Must be >= 0; ``0`` returns an
                empty list.

        Returns:
            Up to ``k`` ``(id, score)`` pairs sorted by fused score descending,
            ties broken by id ascending.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return []
        vector_hits = self._vector_search(text, k)
        keyword_hits = self._keyword_search(text, k)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self._k)
        return fused[:k]
