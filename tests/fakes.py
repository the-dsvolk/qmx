"""Deterministic test doubles — the CI/unit tier of the config seam (no model backend)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

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


class FakeChat:
    """Scripted :class:`~qmx.chat.ChatModel` double — no backend.

    ``extractions`` is a queue of candidate-lists (one per extract pass; reused when exhausted);
    ``decisions`` a queue of consolidate decisions (default ``new``). The system prompt tells the
    two passes apart (extract prompts contain "distill").
    """

    def __init__(
        self, extractions: list[list[dict]] | None = None, decisions: list[dict] | None = None
    ) -> None:
        self._extractions = list(extractions or [])
        self._decisions = list(decisions or [])

    def complete_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        if "distill" in system:  # EXTRACT_SYSTEM
            learnings = self._extractions.pop(0) if self._extractions else []
            return {"learnings": learnings}
        return self._decisions.pop(0) if self._decisions else {"action": "new"}


def build_index(tmp_path: Path, embedder: FakeEmbedder, files: dict[str, str]):
    """Write ``files`` under ``tmp_path``, index them, and return a Settings pointing at the DB.

    ``ollama_url`` is a dead port so ``QmxService`` health checks resolve to False.
    """
    from qmx.config import Settings
    from qmx.index import index_paths
    from qmx.store import Store

    for name, content in files.items():
        (tmp_path / name).write_text(content)
    db = tmp_path / "index.db"
    with Store.open(db, embedder.dim, "fake") as store:
        index_paths([tmp_path], store, embedder)
    return Settings(
        db_path=db,
        embed_dim=embedder.dim,
        embed_model="fake",
        ollama_url="http://127.0.0.1:9",
    )
