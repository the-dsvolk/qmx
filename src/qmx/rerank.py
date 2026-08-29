"""Reranking — reorder the fused top-k with a cross-encoder.

Default is RRF-only (``NoOpReranker``). When ``rerank_url`` is set, :class:`HttpReranker`
calls a Cohere-style ``/v1/rerank`` endpoint — in our deployment that's **llama.cpp `llama-server
--reranking` serving Qwen3-Reranker on the Spark GPU** (see ``plan/qmx-ml-notes.md`` TD-1). It's a
thin HTTP client and **fails soft**: if the rerank server is unreachable, the RRF order is kept.

Failing soft is not enough when the Spark is off — waiting out a 30s read timeout on every query
is its own outage. So the connect phase gets a short budget (a host that is not there says so
immediately) while a real rerank keeps the full read budget, and one failure parks the server for
``CIRCUIT_COOLDOWN`` seconds so a resident server does not re-pay even that.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from qmx.config import Settings
    from qmx.search import RankedHit

log = logging.getLogger("qmx.rerank")

# Reranking a full candidate pool on the GPU can legitimately take seconds, so the read budget
# stays generous; reaching an absent host should not, so the connect budget is short.
CONNECT_TIMEOUT = 2.0

# How long an unreachable rerank server is skipped outright before the next attempt.
CIRCUIT_COOLDOWN = 30.0


@runtime_checkable
class Reranker(Protocol):
    """Reorders fused candidates by relevance to the query."""

    def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        """Return ``hits`` reordered best-first (may also drop/trim)."""
        ...


class NoOpReranker:
    """Passthrough — keeps the RRF order."""

    def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        return hits


class HttpReranker:
    """Cross-encoder reranker via a Cohere-style ``/v1/rerank`` HTTP endpoint.

    Works with llama.cpp's ``llama-server --reranking`` (Qwen3-Reranker): POSTs ``{query,
    documents}`` and reads ``{"results": [{"index", "relevance_score"}]}``. On any transport/parse
    error it returns the input order unchanged (reranking is a refinement, never a hard dependency).
    """

    def __init__(
        self,
        url: str,
        model: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        cooldown: float = CIRCUIT_COOLDOWN,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        base = url.rstrip("/")
        self._endpoint = base if base.endswith("rerank") else f"{base}/v1/rerank"
        self._model = model
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)
        )
        self._cooldown = cooldown
        self._clock = clock
        self._open_until = 0.0

    def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        if not hits:
            return hits
        if self._clock() < self._open_until:  # server known down; don't wait on it again
            return hits
        payload: dict = {"query": query, "documents": [h.hit.text for h in hits]}
        if self._model:
            payload["model"] = self._model
        try:
            resp = self._client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            results = resp.json()["results"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("rerank unavailable, keeping RRF order: %s", exc)
            self._open_until = self._clock() + self._cooldown
            return hits
        self._open_until = 0.0

        ordered: list[RankedHit] = []
        seen: set[int] = set()
        for r in sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True):
            i = r.get("index")
            if isinstance(i, int) and 0 <= i < len(hits) and i not in seen:
                hits[i].score = float(r["relevance_score"])
                ordered.append(hits[i])
                seen.add(i)
        ordered.extend(h for j, h in enumerate(hits) if j not in seen)  # keep any not returned
        return ordered


def make_reranker(settings: Settings) -> Reranker | None:
    """Build a reranker: :class:`HttpReranker` when ``rerank_url`` is set, else ``None``."""
    if getattr(settings, "rerank_url", ""):
        return HttpReranker(settings.rerank_url, model=settings.rerank_model)
    return None
