"""Hybrid search: vector (cosine) + BM25, fused with Reciprocal Rank Fusion.

Phase 1 delivers vector + BM25 -> RRF. The optional Qwen3-Reranker re-orders the fused top-k in
Phase 3 (``plan/qmx-plan.md``); the ``rerank`` seam is left for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qmx.embed import Embedder
from qmx.store import SearchHit, Store

RRF_K = 60  # standard RRF damping constant


@dataclass(slots=True)
class RankedHit:
    """A fused result: the underlying hit plus its RRF score (higher = better)."""

    hit: SearchHit
    score: float


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Fuse ranked id-lists into ``{id: score}`` via RRF (``sum 1/(k + rank)``)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    k: int = 10,
    kind: str | None = None,
    pool: int | None = None,
) -> list[RankedHit]:
    """Run vector + BM25 over ``store`` and return the RRF-fused top-``k``.

    ``pool`` is how many candidates each arm contributes before fusion (default ``max(4k, 20)``).
    """
    pool = pool or max(4 * k, 20)
    [query_vec] = embedder.embed([query])

    vec_hits = store.search_vec(query_vec, k=pool, kind=kind)
    fts_hits = store.search_fts(query, k=pool, kind=kind)

    by_id: dict[int, SearchHit] = {h.chunk_id: h for h in vec_hits}
    for h in fts_hits:
        by_id.setdefault(h.chunk_id, h)

    fused = reciprocal_rank_fusion([[h.chunk_id for h in vec_hits], [h.chunk_id for h in fts_hits]])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [RankedHit(hit=by_id[cid], score=score) for cid, score in ranked]
