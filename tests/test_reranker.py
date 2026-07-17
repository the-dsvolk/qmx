"""HttpReranker (Cohere-style /v1/rerank client), make_reranker factory, and config."""

from __future__ import annotations

from pathlib import Path

import httpx

from qmx.config import Settings
from qmx.rerank import HttpReranker, NoOpReranker, make_reranker
from qmx.search import RankedHit
from qmx.store import SearchHit


def _hits(*texts: str) -> list[RankedHit]:
    return [
        RankedHit(hit=SearchHit(i, 1, "code", f"f{i}.py", t, 0.0), score=1.0 / (i + 1))
        for i, t in enumerate(texts)
    ]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_reranker_reorders_by_relevance():
    # server says doc index 2 is most relevant, then 0, then 1
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/v1/rerank")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.10},
                    {"index": 1, "relevance_score": 0.01},
                    {"index": 2, "relevance_score": 0.99},
                ]
            },
        )

    rr = HttpReranker("http://spark:8081", client=_client(handler))
    out = rr.rerank("q", _hits("alpha", "beta", "gamma"))
    assert [h.hit.text for h in out] == ["gamma", "alpha", "beta"]
    assert out[0].score == 0.99  # score updated to the rerank relevance


def test_http_reranker_fails_soft_on_error():
    def boom(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    original = _hits("a", "b", "c")
    out = HttpReranker("http://spark:8081", client=_client(boom)).rerank("q", list(original))
    assert [h.hit.text for h in out] == ["a", "b", "c"]  # unchanged RRF order


def test_http_reranker_endpoint_normalization():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        return httpx.Response(200, json={"results": []})

    HttpReranker("http://spark:8081/", client=_client(handler)).rerank("q", _hits("x"))
    assert seen["path"] == "/v1/rerank"
    # an explicit rerank path is respected, not double-suffixed
    HttpReranker("http://spark:8081/rerank", client=_client(handler)).rerank("q", _hits("x"))
    assert seen["path"] == "/rerank"


def test_make_reranker_from_settings():
    off = Settings.load(config_path=Path("/does/not/exist"), env={})
    assert make_reranker(off) is None  # rerank_url unset -> disabled
    on = Settings.load(
        config_path=Path("/does/not/exist"), env={"QMX_RERANK_URL": "http://spark-0e81.local:8081"}
    )
    assert isinstance(make_reranker(on), HttpReranker)
    assert on.rerank_url == "http://spark-0e81.local:8081"


def test_noop_still_passthrough():
    h = _hits("a", "b")
    assert NoOpReranker().rerank("q", h) is h
