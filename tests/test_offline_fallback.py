"""Reads keep working with no model backend — the Spark is a LAN box that is sometimes off.

The index is local SQLite, so BM25 can always answer; only the vector arm needs Ollama. These
tests pin that (a) ``search`` degrades instead of raising, (b) the circuit breaker turns a missing
backend into one timeout rather than one per query, and (c) the service says the results are
keyword-only.
"""

from __future__ import annotations

import httpx
import pytest

from qmx.embed import CircuitBreakerEmbedder, EmbedBackendError
from qmx.rerank import HttpReranker
from qmx.search import RankedHit, search
from qmx.service import DEGRADED_NOTE, QmxService
from qmx.store import SearchHit, Store
from tests.fakes import FakeEmbedder, build_index

FILES = {
    "net.py": (
        "import time\n\n\n"
        "def retry_with_backoff(func, attempts=5):\n"
        '    """Retry a callable with exponential backoff between failed attempts."""\n'
        "    return func()\n"
    ),
    "math_utils.py": "def add(a, b):\n    return a + b\n",
}


class DownEmbedder:
    """An embedding backend that is simply not there. Counts how often it was actually asked."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise EmbedBackendError("connection refused")


class FlakyEmbedder(FakeEmbedder):
    """Fails while ``down`` is set, embeds normally once it is cleared."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim)
        self.down = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.down:
            raise EmbedBackendError("connection refused")
        return super().embed(texts)


class FakeClock:
    """Manually advanced monotonic clock, so cooldown tests need no sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def settings(tmp_path):
    return build_index(tmp_path, FakeEmbedder(dim=64), FILES)


def test_search_falls_back_to_bm25_when_embedding_backend_is_down(settings):
    with Store.open(settings.db_path, 64, "fake") as store:
        hits = search(store, DownEmbedder(), "backoff", k=5)
    assert hits, "BM25 is local; a missing embedding backend must not empty the results"
    assert any("net.py" in (h.hit.path or "") for h in hits)


def test_search_still_uses_both_arms_when_the_backend_is_up(settings):
    with Store.open(settings.db_path, 64, "fake") as store:
        hits = search(store, FakeEmbedder(dim=64), "retry a failed request with backoff", k=5)
    assert hits[0].hit.symbol == "retry_with_backoff"


def test_circuit_breaker_asks_the_backend_once_then_fails_fast():
    inner = DownEmbedder()
    clock = FakeClock()
    embedder = CircuitBreakerEmbedder(inner, cooldown=30.0, clock=clock)

    with pytest.raises(EmbedBackendError):
        embedder.embed(["one"])
    assert embedder.degraded
    for _ in range(3):
        with pytest.raises(EmbedBackendError):
            embedder.embed(["again"])
    assert inner.calls == 1, "queries during the cooldown must not re-walk the retry ladder"

    clock.now += 31.0
    assert not embedder.degraded
    with pytest.raises(EmbedBackendError):
        embedder.embed(["after cooldown"])
    assert inner.calls == 2


def test_circuit_closes_again_once_the_backend_returns():
    inner = FlakyEmbedder(dim=64)
    clock = FakeClock()
    embedder = CircuitBreakerEmbedder(inner, cooldown=30.0, clock=clock)

    with pytest.raises(EmbedBackendError):
        embedder.embed(["one"])
    assert embedder.degraded

    inner.down = False
    clock.now += 31.0
    assert len(embedder.embed(["one"])) == 1
    assert not embedder.degraded


def test_service_query_answers_and_flags_the_results_as_keyword_only(settings):
    service = QmxService(settings, DownEmbedder())
    hits = service.query("backoff", k=5)
    assert hits
    assert all(h["degraded"] == DEGRADED_NOTE for h in hits)


def test_service_query_has_no_degraded_marker_when_the_backend_is_up(settings):
    hits = QmxService(settings, FakeEmbedder(dim=64)).query("backoff", k=5)
    assert hits and all("degraded" not in h for h in hits)


class CountingReranker:
    """Records whether search bothered to call it at all."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, hits):
        self.calls += 1
        return hits


def test_degraded_search_skips_the_reranker_too(settings):
    """It lives on the same box as the embedding backend; waiting on it would undo the point."""
    reranker = CountingReranker()
    with Store.open(settings.db_path, 64, "fake") as store:
        search(store, DownEmbedder(), "backoff", k=5, reranker=reranker)
    assert reranker.calls == 0

    with Store.open(settings.db_path, 64, "fake") as store:
        search(store, FakeEmbedder(dim=64), "backoff", k=5, reranker=reranker)
    assert reranker.calls == 1, "with the backend up the reranker still runs"


def _one_hit(text="alpha"):
    return [
        RankedHit(
            hit=SearchHit(
                chunk_id=1,
                doc_id=1,
                kind="code",
                path="a.py",
                start_line=1,
                end_line=2,
                symbol=None,
                text=text,
                distance=0.1,
            ),
            score=0.5,
        )
    ]


def test_unreachable_rerank_server_is_asked_once_then_skipped():
    calls = {"n": 0}

    def boom(request):
        calls["n"] += 1
        raise httpx.ConnectTimeout("no route to host")

    clock = FakeClock()
    rr = HttpReranker(
        "http://spark:8081",
        client=httpx.Client(transport=httpx.MockTransport(boom)),
        cooldown=30.0,
        clock=clock,
    )
    assert rr.rerank("q", _one_hit()) == _one_hit()  # fails soft, RRF order kept
    for _ in range(3):
        rr.rerank("q", _one_hit())
    assert calls["n"] == 1, "a known-down rerank server must not be waited on again"

    clock.now += 31.0
    rr.rerank("q", _one_hit())
    assert calls["n"] == 2


def test_service_status_reports_degraded(settings, monkeypatch):
    monkeypatch.setattr("qmx.service.ping_ollama", lambda *a, **kw: False)
    service = QmxService(settings, DownEmbedder())
    assert service.status()["degraded"] is False  # nothing has failed yet
    service.query("backoff", k=5)
    status = service.status()
    assert status["degraded"] is True
    assert status["ollama_ok"] is False
