"""Embeddings — thin HTTP clients behind an :class:`Embedder` protocol.

qmx never loads models in-process (no torch); it POSTs to a model server on the Spark (see
``plan/qmx-deployment.md``): Ollama's ``/api/embed`` (``embed_backend = "ollama"``, the default) or
an OpenAI-compatible ``/v1/embeddings`` server such as vLLM (``"openai"``). The :class:`Embedder`
protocol is the seam that lets tests and CI swap in a deterministic fake with no backend.

Two sides. ``embed`` is the *document* side: indexed chunks, lessons, parked items, and the
doc-vs-doc comparisons (insert-time dedup, similar parked items). ``embed_query`` is the *query*
side, used by search only. Asymmetric models (Nemotron Embed: ``"query: "`` / ``"passage: "``) need
the matching prefix on each side; ``embed_query_prefix`` / ``embed_passage_prefix`` supply them and
default to empty, which is exactly what the Qwen index was built with. :func:`embed_queries` calls
``embed_query`` when an embedder has one and ``embed`` otherwise, so simple fakes keep working.

The backend is not always there (the Spark is a LAN box that can be off or unreachable), so the
read path degrades instead of failing: :class:`CircuitBreakerEmbedder` makes the second and later
attempts fail instantly rather than re-walking the retry ladder, and :func:`qmx.search.search`
falls back to BM25-only when embedding raises.
"""

from __future__ import annotations

import base64
import binascii
import math
import sys
import time
from array import array
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
        """Return one document-side embedding per input text, order-preserving."""
        ...


def embed_queries(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Query-side embeddings: ``embed_query`` when the embedder has one, else plain ``embed``."""
    fn = getattr(embedder, "embed_query", None)
    return fn(texts) if fn is not None else embedder.embed(texts)


def ping_embedder(settings: Settings, timeout: float = 2.0) -> bool:
    """Is the embedding backend reachable right now? One cheap GET, never raises.

    Ollama: ``/api/version`` at ``ollama_url``. OpenAI-compatible: ``/models`` under ``embed_url``.
    """
    if settings.embed_backend == "openai":
        url = f"{settings.embed_url.rstrip('/')}/models"
    else:
        url = f"{settings.ollama_url.rstrip('/')}/api/version"
    try:
        return httpx.get(url, timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


# Kept for callers that predate the backend switch; it now pings whichever backend is configured.
ping_ollama = ping_embedder


class _HttpEmbedder:
    """Shared batching, prefixing and retry ladder; subclasses do one request + response shape."""

    _label = "embed"

    def __init__(
        self,
        settings: Settings,
        base_url: str,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._batch_size = settings.embed_batch_size
        self._query_prefix = settings.embed_query_prefix
        self._passage_prefix = settings.embed_passage_prefix
        self._max_retries = settings.max_retries if max_retries is None else max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        timeout = (
            settings.request_timeout
            if connect_timeout is None
            else httpx.Timeout(settings.request_timeout, connect=connect_timeout)
        )
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Document side: ``embed_passage_prefix`` + text."""
        return self._embed_all([self._passage_prefix + t for t in texts])

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Query side: ``embed_query_prefix`` + text."""
        return self._embed_all([self._query_prefix + t for t in texts])

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise EmbedBackendError(
                    f"{self._label} returned {len(vectors)} embeddings for {len(batch)} inputs"
                )
            out.extend(vectors)
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:  # network, timeout, bad status, bad JSON
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._base_delay * (2**attempt))
        plural = "" if self._max_retries == 1 else "s"
        raise EmbedBackendError(
            f"{self._label} failed after {self._max_retries} attempt{plural}: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OllamaEmbedder(_HttpEmbedder):
    """Batched, retrying client for Ollama's ``/api/embed`` endpoint."""

    _label = "Ollama embed"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        super().__init__(settings, settings.ollama_url, client, max_retries, connect_timeout)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = self._post_with_retry("/api/embed", {"model": self._model, "input": batch})
        try:
            vectors = body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise EmbedBackendError(f"Ollama embed: unexpected response body: {exc}") from exc
        for vec in vectors:
            if len(vec) != self._dim:
                raise EmbedBackendError(
                    f"embedding dim {len(vec)} != configured embed_dim {self._dim} "
                    f"(model {self._model!r}); fix QMX_EMBED_DIM"
                )
        return vectors


class OpenAIEmbedder(_HttpEmbedder):
    """Batched, retrying client for an OpenAI-compatible ``/v1/embeddings`` server (vLLM).

    Vectors come back base64-encoded (a third of the JSON-float size). A model wider than
    ``embed_dim`` (Matryoshka-trained, e.g. Nemotron-3-Embed-1B's 2048) is cut to its first
    ``embed_dim`` values and re-normalised to unit length; a narrower one is an error.
    ``embed_max_tokens`` > 0 sends vLLM's ``truncate_prompt_tokens`` so an over-long chunk is cut
    server-side instead of rejected.
    """

    _label = "OpenAI-compatible embed"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        if not settings.embed_url and client is None:
            raise ValueError('embed_backend = "openai" needs embed_url (base URL incl. /v1)')
        super().__init__(settings, settings.embed_url, client, max_retries, connect_timeout)
        self._max_tokens = settings.embed_max_tokens

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict = {"model": self._model, "input": batch, "encoding_format": "base64"}
        if self._max_tokens > 0:
            payload["truncate_prompt_tokens"] = self._max_tokens
        body = self._post_with_retry("/embeddings", payload)
        try:
            data = sorted(body["data"], key=lambda d: d["index"])
            raw = [_decode_base64_f32(d["embedding"]) for d in data]
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise EmbedBackendError(f"{self._label}: unexpected response body: {exc}") from exc
        return [self._fit(vec) for vec in raw]

    def _fit(self, vec: list[float]) -> list[float]:
        if len(vec) == self._dim:
            return vec
        if len(vec) < self._dim:
            raise EmbedBackendError(
                f"embedding dim {len(vec)} < configured embed_dim {self._dim} "
                f"(model {self._model!r}); fix QMX_EMBED_DIM"
            )
        return _unit(vec[: self._dim])


def _decode_base64_f32(encoded: str) -> list[float]:
    """Little-endian float32 vector from the OpenAI base64 encoding."""
    buf = array("f")
    buf.frombytes(base64.b64decode(encoded, validate=True))
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tolist()


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0.0 else [x / norm for x in vec]


EMBED_BACKENDS = ("ollama", "openai")


def make_embedder(
    settings: Settings,
    *,
    max_retries: int | None = None,
    connect_timeout: float | None = None,
) -> OllamaEmbedder | OpenAIEmbedder:
    """The embedding client ``settings.embed_backend`` selects (``ollama`` default, ``openai``)."""
    if settings.embed_backend == "ollama":
        return OllamaEmbedder(settings, max_retries=max_retries, connect_timeout=connect_timeout)
    if settings.embed_backend == "openai":
        return OpenAIEmbedder(settings, max_retries=max_retries, connect_timeout=connect_timeout)
    raise ValueError(
        f"embed_backend must be one of {', '.join(EMBED_BACKENDS)}, got {settings.embed_backend!r}"
    )


def read_embedder(settings: Settings) -> CircuitBreakerEmbedder:
    """The embedder for *read* commands (query / recall / lessons): fail fast, degrade gracefully.

    Three things differ from the indexing embedder: a short connect budget, a single attempt, and
    a circuit so the following queries do not each re-pay even that. Together they turn "the Spark
    is off" into a few seconds once, after which :func:`qmx.search.search` answers BM25-only from
    the local index.
    """
    return CircuitBreakerEmbedder(interactive_embedder(settings), cooldown=CIRCUIT_COOLDOWN)


def interactive_embedder(settings: Settings) -> OllamaEmbedder | OpenAIEmbedder:
    """A plain embedder tuned to give up quickly — for paths a human is waiting on."""
    return make_embedder(
        settings,
        max_retries=INTERACTIVE_MAX_RETRIES,
        connect_timeout=INTERACTIVE_CONNECT_TIMEOUT,
    )


class CircuitBreakerEmbedder:
    """Fail-fast wrapper around an :class:`Embedder`, so a missing backend costs one timeout.

    The HTTP embedders retry ``max_retries`` times with exponential backoff — right for
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
        return self._guarded(self._inner.embed, texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._guarded(lambda t: embed_queries(self._inner, t), texts)

    def _guarded(
        self, fn: Callable[[list[str]], list[list[float]]], texts: list[str]
    ) -> list[list[float]]:
        if self.degraded:
            raise EmbedBackendError(
                "embedding backend marked down for another "
                f"{self._open_until - self._clock():.0f}s (not retrying yet)"
            )
        try:
            vectors = fn(texts)
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
