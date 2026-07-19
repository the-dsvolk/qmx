"""Hybrid search: vector (cosine) + BM25, fused with Reciprocal Rank Fusion.

Ranking is RRF over the vector and BM25 arms. An optional :class:`~qmx.rerank.Reranker` reorders
the fused top candidates when ``rerank_url`` is set (a Qwen3-Reranker via llama.cpp on the Spark);
it is off by default (RRF-only) and fails soft to RRF order — see ``plan/qmx-ml-notes.md`` TD-1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qmx.embed import Embedder
from qmx.store import SearchHit, Store

if TYPE_CHECKING:
    from qmx.rerank import Reranker

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
    reranker: Reranker | None = None,
    rerank_pool: int = 40,
) -> list[RankedHit]:
    """Run vector + BM25 over ``store`` and return the RRF-fused top-``k``.

    ``pool`` is how many candidates each arm contributes before fusion (default ``max(4k, 20)``).
    If a ``reranker`` is given, the RRF top ``rerank_pool`` candidates are reranked and trimmed to
    ``k``; otherwise the RRF top-``k`` is returned (see ``plan/qmx-ml-notes.md`` TD-1).
    """
    pool = pool or max(4 * k, 20)
    [query_vec] = embedder.embed([query])

    vec_hits = store.search_vec(query_vec, k=pool, kind=kind)
    fts_hits = store.search_fts(query, k=pool, kind=kind)

    by_id: dict[int, SearchHit] = {h.chunk_id: h for h in vec_hits}
    for h in fts_hits:
        by_id.setdefault(h.chunk_id, h)

    fused = reciprocal_rank_fusion([[h.chunk_id for h in vec_hits], [h.chunk_id for h in fts_hits]])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    # Rerank a wider candidate pool (then trim to k); without a reranker, RRF top-k is final.
    take = max(k, rerank_pool) if reranker is not None else k
    hits = [RankedHit(hit=by_id[cid], score=score) for cid, score in ranked[:take]]
    if reranker is not None:
        hits = reranker.rerank(query, hits)
    return hits[:k]
