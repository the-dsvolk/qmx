"""Service layer — the read API the MCP server (and tests) call.

One place that owns "open the store, run a search, shape a JSON-friendly result", so the MCP tools
stay thin. Each call opens a short-lived store connection (SQLite WAL handles concurrent readers);
the embedder/HTTP client is shared for the service's lifetime.
"""

from __future__ import annotations

import httpx

from qmx.config import Settings
from qmx.embed import Embedder, OllamaEmbedder
from qmx.learnings import (
    add_learning,
    deprecate_learning,
    learning_to_dict,
    lessons,
    restore_learning,
    update_learning,
)
from qmx.rerank import make_reranker
from qmx.search import search
from qmx.store import KEEP, Learning, SearchHit, Store

MAX_TEXT_CHARS = 4000  # cap chunk text returned to an agent so results stay compact


class QmxService:
    """Operations backing the MCP tools: ``query`` / ``recall`` / ``lessons`` / ``get`` / ``status``
    reads, and the ``add_learning`` / ``update_learning`` / ``deprecate_learning`` /
    ``restore_learning`` writes over the index."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self._settings = settings
        self._embedder = embedder if embedder is not None else OllamaEmbedder(settings)
        self._reranker = make_reranker(settings)  # None unless rerank_url is configured

    def _store(self) -> Store:
        return Store.open(
            self._settings.db_path, self._settings.embed_dim, self._settings.embed_model
        )

    def query(self, text: str, k: int = 5, kind: str | None = None) -> list[dict]:
        """Hybrid (vector + BM25 -> RRF, optional rerank) search; JSON-friendly hits."""
        with self._store() as store:
            results = search(store, self._embedder, text, k=k, kind=kind, reranker=self._reranker)
            return [_hit_dict(r.hit, score=r.score) for r in results]

    def recall(self, text: str, k: int = 5) -> list[dict]:
        """Search **chat** memory only (``kind='chat'``) — past Claude Code conversation turns."""
        with self._store() as store:
            results = search(store, self._embedder, text, k=k, kind="chat", reranker=self._reranker)
            return [_hit_dict(r.hit, score=r.score) for r in results]

    def lessons(
        self,
        query: str,
        k: int = 5,
        type: str | None = None,
        scope: str | None = None,
        include_retired: bool = False,
    ) -> list[dict]:
        """Retrieve distilled lessons (``kind='learning'``) by relevance×importance×recency."""
        with self._store() as store:
            return lessons(
                store,
                self._embedder,
                query,
                k=k,
                type=type,
                scope=scope,
                include_retired=include_retired,
                reranker=self._reranker,
            )

    def add_learning(
        self,
        *,
        type: str,
        statement: str,
        topic: str | None = None,
        scope: str | None = None,
        detail: str | None = None,
        importance: float = 0.5,
    ) -> dict:
        """Manually add a lesson (seed / promote-from-review); returns the stored learning."""
        with self._store() as store:
            learning_id = add_learning(
                store,
                self._embedder,
                type=type,
                statement=statement,
                topic=topic,
                scope=scope,
                detail=detail,
                importance=importance,
            )
            learning = store.get_learning(learning_id)
        return learning_to_dict(learning)

    def update_learning(
        self,
        learning_id: int,
        *,
        type: str = KEEP,
        topic: str | None = KEEP,
        scope: str | None = KEEP,
        statement: str = KEEP,
        detail: str | None = KEEP,
        importance: float = KEEP,
    ) -> dict | None:
        """Fix a lesson in place (statement/detail/type/topic/scope/importance)."""
        with self._store() as store:
            learning = update_learning(
                store,
                self._embedder,
                learning_id,
                type=type,
                topic=topic,
                scope=scope,
                statement=statement,
                detail=detail,
                importance=importance,
            )
        return _learning_or_none(learning)

    def deprecate_learning(
        self, learning_id: int, *, reason: str | None = None, superseded_by: int | None = None
    ) -> dict | None:
        """Soft-retire a lesson — hidden from retrieval, kept with a breadcrumb."""
        with self._store() as store:
            learning = deprecate_learning(
                store, learning_id, reason=reason, superseded_by=superseded_by
            )
        return _learning_or_none(learning)

    def restore_learning(self, learning_id: int) -> dict | None:
        """Undo a retirement — the lesson is live (and retrievable) again."""
        with self._store() as store:
            learning = restore_learning(store, learning_id)
        return _learning_or_none(learning)

    def get(self, chunk_id: int) -> dict | None:
        """Full text + location for one chunk, or ``None`` if it is gone/tombstoned."""
        with self._store() as store:
            hit = store.get_chunk(chunk_id)
        return None if hit is None else _hit_dict(hit, score=None, full=True)

    def status(self) -> dict:
        """Index stats + backend health, for ops and the MCP ``status`` tool."""
        with self._store() as store:
            index = store.index_stats()
        return {
            "index": index,
            "embed_model": self._settings.embed_model,
            "ollama_url": self._settings.ollama_url,
            "ollama_ok": self._ping(),
        }

    def _ping(self) -> bool:
        try:
            resp = httpx.get(f"{self._settings.ollama_url.rstrip('/')}/api/version", timeout=2.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _learning_or_none(learning: Learning | None) -> dict | None:
    return None if learning is None else learning_to_dict(learning)


def _hit_dict(hit: SearchHit, *, score: float | None, full: bool = False) -> dict:
    text = hit.text if full else hit.text[:MAX_TEXT_CHARS]
    out = {
        "chunk_id": hit.chunk_id,
        "kind": hit.kind,
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "symbol": hit.symbol,
        "text": text,
    }
    if score is not None:
        out["score"] = round(score, 6)
    return out
