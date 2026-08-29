"""Embeddings — a thin Ollama HTTP client behind an :class:`Embedder` protocol.

qmx never loads models in-process (no torch); it POSTs to Ollama, which runs on the Spark in prod
(see ``plan/qmx-deployment.md``). The :class:`Embedder` protocol is the seam that lets tests and CI
swap in a deterministic fake with no backend.

The backend is not always there (the Spark is a LAN box that can be off or unreachable), so the
read path degrades instead of failing: :class:`CircuitBreakerEmbedder` makes the second and later
attempts fail instantly rather than re-walking the retry ladder, and :func:`qmx.search.search`
falls back to BM25-only when embedding raises.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import httpx

from qmx.config import Settings

# Latency-sensitive callers (a query, the Stop hook) try once. On a LAN a reachable backend
# connects in milliseconds, so a failure here means "absent", not "busy" — and an absent host is
# expensive to ask twice (see ``INTERACTIVE_CONNECT_TIMEOUT``). Recovery is the cooldown below,
# not a retry. Indexing keeps ``settings.max_retries``: there, waiting out a blip beats a stale
# index.
INTERACTIVE_MAX_RETRIES = 1

# How long a failed backend stays marked down before the read path probes it again.
CIRCUIT_COOLDOWN = 30.0

# ``request_timeout`` is sized for embedding a batch and applies to the connect phase too, which is
# the wrong budget for reaching an absent host. Note this only caps the TCP connect: resolving an
# offline ``*.local`` name costs a further ~5s of mDNS that no HTTP timeout covers, which is the
# real reason these paths try exactly once.
INTERACTIVE_CONNECT_TIMEOUT = 2.0


class EmbedBackendError(RuntimeError):
    """Raised when the embedding backend is unreachable after all retries."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-width vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, order-preserving."""
        ...


def ping_ollama(settings: Settings, timeout: float = 2.0) -> bool:
    """Is the Ollama backend reachable right now? Cheap ``GET /api/version``, never raises."""
    try:
        resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/version", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


class OllamaEmbedder:
    """Batched, retrying client for Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._batch_size = settings.embed_batch_size
        self._max_retries = settings.max_retries if max_retries is None else max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        timeout = (
            settings.request_timeout
            if connect_timeout is None
            else httpx.Timeout(settings.request_timeout, connect=connect_timeout)
        )
        self._client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=timeout,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": batch}
        vectors = self._post_with_retry("/api/embed", payload)
        if len(vectors) != len(batch):
            raise EmbedBackendError(
                f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs"
            )
        for vec in vectors:
            if len(vec) != self._dim:
                raise EmbedBackendError(
                    f"embedding dim {len(vec)} != configured embed_dim {self._dim} "
                    f"(model {self._model!r}); fix QMX_EMBED_DIM"
                )
        return vectors

    def _post_with_retry(self, path: str, payload: dict) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except (httpx.HTTPError, KeyError) as exc:  # network, timeout, bad status, bad body
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._base_delay * (2**attempt))
        plural = "" if self._max_retries == 1 else "s"
        raise EmbedBackendError(
            f"Ollama embed failed after {self._max_retries} attempt{plural}: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_embedder(settings: Settings) -> CircuitBreakerEmbedder:
    """The embedder for *read* commands (query / recall / lessons): fail fast, degrade gracefully.

    Three things differ from the indexing embedder: a short connect budget, a single attempt, and
    a circuit so the following queries do not each re-pay even that. Together they turn "the Spark
    is off" into a few seconds once, after which :func:`qmx.search.search` answers BM25-only from
    the local index.
    """
    return CircuitBreakerEmbedder(interactive_embedder(settings), cooldown=CIRCUIT_COOLDOWN)


def interactive_embedder(settings: Settings) -> OllamaEmbedder:
    """A plain embedder tuned to give up quickly — for paths a human is waiting on."""
    return OllamaEmbedder(
        settings,
        max_retries=INTERACTIVE_MAX_RETRIES,
        connect_timeout=INTERACTIVE_CONNECT_TIMEOUT,
    )


class CircuitBreakerEmbedder:
    """Fail-fast wrapper around an :class:`Embedder`, so a missing backend costs one timeout.

    :class:`OllamaEmbedder` retries ``max_retries`` times with exponential backoff — right for
    indexing, far too slow for an interactive query when the Spark is simply off. After one failure
    this marks the backend down for ``cooldown`` seconds and raises :class:`EmbedBackendError`
    immediately; a later success closes the circuit again. ``degraded`` lets callers tell an agent
    that its results came back keyword-only.
    """

    def __init__(
        self,
        inner: Embedder,
        cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._cooldown = cooldown
        self._clock = clock
        self._open_until = 0.0

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def degraded(self) -> bool:
        """True while the circuit is open — the backend failed within the last ``cooldown``."""
        return self._clock() < self._open_until

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.degraded:
            raise EmbedBackendError(
                "embedding backend marked down for another "
                f"{self._open_until - self._clock():.0f}s (not retrying yet)"
            )
        try:
            vectors = self._inner.embed(texts)
        except EmbedBackendError:
            self._open_until = self._clock() + self._cooldown
            raise
        self._open_until = 0.0
        return vectors

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> CircuitBreakerEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
