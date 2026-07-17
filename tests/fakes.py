"""Deterministic test doubles — the CI/unit tier of the config seam (no model backend)."""

from __future__ import annotations

import hashlib
import re

# Drop punctuation-glued tokens down to word cores, and ignore very common/short words so that
# cosine overlap reflects *content* words — a crude but realistic stand-in for real preprocessing.
_WORD = re.compile(r"[a-z0-9]+")
# fmt: off
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "that",
    "this", "for", "with", "how", "do", "def", "return", "self",
})
# fmt: on


class FakeEmbedder:
    """Bag-of-content-words hashing embedder: deterministic, no backend.

    Identical text -> identical vector (cosine distance ~0); texts sharing content words land
    closer than disjoint ones. Enough to exercise store + cosine top-k ordering.
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
        for tok in _WORD.findall(text.lower()):
            if len(tok) < 3 or tok in _STOP:
                continue
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self._dim] += 1.0
        if not any(v):  # nothing content-bearing -> a fixed non-zero vector (cosine needs non-zero)
            v[0] = 1.0
        return v
