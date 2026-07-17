"""Deterministic test doubles — the CI/unit tier of the config seam (no model backend)."""

from __future__ import annotations

import hashlib


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, no backend.

    Identical text -> identical vector (cosine distance ~0); texts sharing words land closer than
    disjoint ones. Enough to exercise store + cosine top-k ordering.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in text.lower().split():
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self._dim] += 1.0
        if not any(v):  # empty/whitespace text -> a fixed non-zero vector (cosine needs non-zero)
            v[0] = 1.0
        return v
