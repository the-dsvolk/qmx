"""Rerank seam — no-op default and that search() applies a supplied reranker."""

from __future__ import annotations

import pytest

from qmx.rerank import NoOpReranker
from qmx.search import RankedHit, search
from qmx.store import SearchHit, Store
from tests.fakes import FakeEmbedder, build_index


def _hit(cid: int) -> RankedHit:
    return RankedHit(hit=SearchHit(cid, 1, "code", "p.py", "t", 0.0), score=1.0 / cid)


def test_noop_reranker_is_passthrough():
    hits = [_hit(1), _hit(2)]
    assert NoOpReranker().rerank("q", hits) is hits


class _ReverseReranker:
    def rerank(self, query, hits):
        return list(reversed(hits))


@pytest.fixture
def store_and_embedder(tmp_path):
    embedder = FakeEmbedder(dim=64)
    settings = build_index(
        tmp_path,
        embedder,
        {
            "a.py": "def alpha_vector_search():\n    return 1\n",
            "b.py": "def beta_vector_search():\n    return 2\n",
        },
    )
    store = Store.open(settings.db_path, embedder.dim, "fake")
    yield store, embedder
    store.close()


def test_search_applies_reranker(store_and_embedder):
    store, embedder = store_and_embedder
    base = search(store, embedder, "vector search", k=2)
    assert len(base) == 2
    reranked = search(store, embedder, "vector search", k=2, reranker=_ReverseReranker())
    assert [h.hit.chunk_id for h in reranked] == [h.hit.chunk_id for h in reversed(base)]
