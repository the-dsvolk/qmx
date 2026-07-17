"""Reciprocal Rank Fusion unit tests."""

from __future__ import annotations

from qmx.search import reciprocal_rank_fusion


def test_rrf_rewards_top_ranks():
    scores = reciprocal_rank_fusion([[1, 2, 3]], k=60)
    assert scores[1] > scores[2] > scores[3]


def test_rrf_sums_across_rankings():
    # id 1 is top in BOTH lists; every other id appears in just one.
    scores = reciprocal_rank_fusion([[1, 2, 3], [1, 5, 6]], k=60)
    assert scores[1] == max(scores.values())
    assert scores[1] > scores[2]  # agreement across arms wins over a single strong hit


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}
