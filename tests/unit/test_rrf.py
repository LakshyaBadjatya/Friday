"""Unit tests for ``friday.memory.rrf``.

Cover the pure :func:`reciprocal_rank_fusion` (scoring, ordering, ties,
duplicate handling, ``k`` validation) and :class:`HybridRecall`, which fuses an
injected vector ranker and an injected keyword ranker via RRF.
"""

from __future__ import annotations

import pytest

from friday.memory.rrf import HybridRecall, reciprocal_rank_fusion


def _ids(fused: list[tuple[str, float]]) -> list[str]:
    return [item_id for item_id, _ in fused]


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert _ids(fused) == ["a", "b", "c"]


def test_single_ranking_scores_match_formula() -> None:
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1.0 / 60)
    assert scores["b"] == pytest.approx(1.0 / 61)


def test_item_in_both_rankings_beats_item_in_one() -> None:
    # "b" appears in both rankings; "a" and "c" each appear once near the top.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert _ids(fused)[0] == "b"


def test_fusion_sums_contributions_across_rankings() -> None:
    fused = reciprocal_rank_fusion([["x"], ["x"]], k=60)
    scores = dict(fused)
    assert scores["x"] == pytest.approx(2.0 / 60)


def test_orders_by_score_descending() -> None:
    # "shared" is rank 0 in both -> highest; "only_a"/"only_b" trail it.
    fused = reciprocal_rank_fusion(
        [["shared", "only_a"], ["shared", "only_b"]]
    )
    assert _ids(fused)[0] == "shared"
    assert set(_ids(fused)[1:]) == {"only_a", "only_b"}


def test_ties_broken_by_id_ascending() -> None:
    # Both ids are rank 0 in their own single-ranking -> equal score.
    fused = reciprocal_rank_fusion([["b"], ["a"]])
    assert _ids(fused) == ["a", "b"]


def test_duplicate_id_within_ranking_counts_best_position_only() -> None:
    # "a" appears at rank 0 and rank 2 in the same ranking; only rank 0 counts.
    fused = reciprocal_rank_fusion([["a", "b", "a"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1.0 / 60)


def test_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_larger_k_flattens_top_rank_advantage() -> None:
    high_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    low_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    # The relative gap between rank-0 and rank-1 shrinks as k grows.
    assert (high_k["a"] - high_k["b"]) < (low_k["a"] - low_k["b"])


def test_hybrid_recall_merges_two_rankings() -> None:
    def vector(text: str, k: int) -> list[str]:
        return ["doc1", "doc2", "doc3"]

    def keyword(text: str, k: int) -> list[str]:
        return ["doc3", "doc2", "doc1"]

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    fused = recall.query("anything", k=3)
    # With k=60: doc1 = 1/60+1/62, doc3 = 1/62+1/60 (tie, top), doc2 = 2/61
    # (lower, since 1/60+1/62 > 2/61). doc1 wins the tie by id ascending.
    assert set(_ids(fused)) == {"doc1", "doc2", "doc3"}
    assert _ids(fused)[0] == "doc1"
    assert _ids(fused)[-1] == "doc2"


def test_hybrid_recall_consensus_top_hit_wins() -> None:
    # "shared" is rank 0 in BOTH rankers; nothing else is, so it must win.
    def vector(text: str, k: int) -> list[str]:
        return ["shared", "v1", "v2"]

    def keyword(text: str, k: int) -> list[str]:
        return ["shared", "k1", "k2"]

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    fused = recall.query("q", k=5)
    assert _ids(fused)[0] == "shared"
    assert set(_ids(fused)) == {"shared", "v1", "v2", "k1", "k2"}


def test_hybrid_recall_passes_query_and_k_to_searchers() -> None:
    calls: list[tuple[str, str, int]] = []

    def vector(text: str, k: int) -> list[str]:
        calls.append(("vector", text, k))
        return ["a"]

    def keyword(text: str, k: int) -> list[str]:
        calls.append(("keyword", text, k))
        return ["a"]

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    recall.query("hello world", k=5)
    assert ("vector", "hello world", 5) in calls
    assert ("keyword", "hello world", 5) in calls


def test_hybrid_recall_truncates_to_k() -> None:
    def vector(text: str, k: int) -> list[str]:
        return ["a", "b", "c", "d"]

    def keyword(text: str, k: int) -> list[str]:
        return ["e", "f", "g", "h"]

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    fused = recall.query("q", k=2)
    assert len(fused) == 2


def test_hybrid_recall_only_vector_hits() -> None:
    def vector(text: str, k: int) -> list[str]:
        return ["a", "b"]

    def keyword(text: str, k: int) -> list[str]:
        return []

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    fused = recall.query("q", k=4)
    assert _ids(fused) == ["a", "b"]


def test_hybrid_recall_zero_k_returns_empty_without_calling() -> None:
    called = False

    def vector(text: str, k: int) -> list[str]:
        nonlocal called
        called = True
        return ["a"]

    def keyword(text: str, k: int) -> list[str]:
        nonlocal called
        called = True
        return ["a"]

    recall = HybridRecall(vector_search=vector, keyword_search=keyword)
    assert recall.query("q", k=0) == []
    assert called is False


def test_hybrid_recall_negative_k_raises() -> None:
    recall = HybridRecall(
        vector_search=lambda text, k: [],
        keyword_search=lambda text, k: [],
    )
    with pytest.raises(ValueError):
        recall.query("q", k=-1)


def test_hybrid_recall_rejects_bad_fusion_k() -> None:
    with pytest.raises(ValueError):
        HybridRecall(
            vector_search=lambda text, k: [],
            keyword_search=lambda text, k: [],
            k=0,
        )
