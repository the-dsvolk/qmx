"""Embeddings — a thin Ollama HTTP client behind an :class:`Embedder` protocol.

qmx never loads models in-process (no torch); it POSTs to Ollama, which runs on the Spark in prod
(see ``plan/qmx-deployment.md``). The :class:`Embedder` protocol is the seam that lets tests and CI
swap in a deterministic fake with no backend.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import httpx

from qmx.config import Settings


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


class OllamaEmbedder:
    """Batched, retrying client for Ollama's ``/api/embed`` endpoint."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._batch_size = settings.embed_batch_size
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=settings.request_timeout,
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
        raise EmbedBackendError(
            f"Ollama embed failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
