"""Service layer — the read API the MCP server (and tests) call.

One place that owns "open the store, run a search, shape a JSON-friendly result", so the MCP tools
stay thin. Each call opens a short-lived store connection (SQLite WAL handles concurrent readers);
the embedder/HTTP client is shared for the service's lifetime.

Reads survive a missing embedding backend: the embedder is wrapped in a
:class:`~qmx.embed.CircuitBreakerEmbedder` so one timeout is enough to mark it down, and
:func:`~qmx.search.search` then answers BM25-only from the local index. Results carry a
``degraded`` note in that case so the calling agent knows they are keyword-matched.
"""

from __future__ import annotations

import re
from pathlib import Path

from qmx.config import Settings
from qmx.embed import (
    CircuitBreakerEmbedder,
    EmbedBackendError,
    Embedder,
    ping_ollama,
    read_embedder,
)
from qmx.learnings import (
    add_learning,
    deprecate_learning,
    learning_to_dict,
    lessons,
    restore_learning,
    update_learning,
)
from qmx.parked import (
    KIND as PARKED_KIND,
)
from qmx.parked import (
    ParkedItem,
    ParkedRepo,
    filter_scope,
    item_id_from_path,
    sync_index,
    try_index,
    unindex_item,
)
from qmx.rerank import make_reranker
from qmx.scope import canonical_repo_key
from qmx.search import search
from qmx.store import KEEP, Learning, SearchHit, Store

MAX_TEXT_CHARS = 4000  # cap chunk text returned to an agent so results stay compact

# Attached to every hit returned while the embedding backend is down, so an agent can weigh the
# results (and, if it matters, re-run the query once `status.ollama_ok` is true again).
DEGRADED_NOTE = "bm25-only: embedding backend unreachable, results are keyword-matched"

# Cosine distance under which another open parked item counts as "similar" (a dedup hint on park).
PARKED_SIMILAR_MAX_DISTANCE = 0.3
PARKED_LIST_STATUSES = ("open", "done", "dropped", "closed", "all")

_MEETING_FOLDER_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_")


class QmxService:
    """Operations backing the MCP tools: ``query`` / ``recall`` / ``lessons`` / ``get`` / ``status``
    reads, the ``add_learning`` / ``update_learning`` / ``deprecate_learning`` /
    ``restore_learning`` writes over the index, and the parked-items lifecycle
    (``park`` / ``list_parked`` / ``unpark`` / ``reopen_parked`` / ``keep_parked`` /
    ``resume_parked``)."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self._settings = settings
        self._embedder = (
            CircuitBreakerEmbedder(embedder) if embedder is not None else read_embedder(settings)
        )
        self._reranker = make_reranker(settings)  # None unless rerank_url is configured

    def _mark(self, rows: list[dict]) -> list[dict]:
        """Tag results keyword-only when the vector arm was skipped, so an agent can tell."""
        if self._embedder.degraded:
            for row in rows:
                row["degraded"] = DEGRADED_NOTE
        return rows

    def _store(self) -> Store:
        return Store.open(
            self._settings.db_path, self._settings.embed_dim, self._settings.embed_model
        )

    def query(self, text: str, k: int = 5, kind: str | None = None) -> list[dict]:
        """Hybrid (vector + BM25 -> RRF, optional rerank) search; JSON-friendly hits."""
        with self._store() as store:
            results = search(store, self._embedder, text, k=k, kind=kind, reranker=self._reranker)
            return self._mark([_hit_dict(r.hit, score=r.score) for r in results])

    def recall(self, text: str, k: int = 5) -> list[dict]:
        """Search **chat** memory only (``kind='chat'``) — past Claude Code conversation turns."""
        with self._store() as store:
            results = search(store, self._embedder, text, k=k, kind="chat", reranker=self._reranker)
            return self._mark([_hit_dict(r.hit, score=r.score) for r in results])

    def meetings(self, text: str, k: int = 5) -> list[dict]:
        """Search **meetings** only (``kind='meeting'``) — Memoir call transcripts, summaries,
        notes and prep. Each hit also names its call (``meeting`` folder, ``date``) and ``file``."""
        with self._store() as store:
            results = search(
                store, self._embedder, text, k=k, kind="meeting", reranker=self._reranker
            )
            return self._mark([_meeting_hit_dict(r.hit, score=r.score) for r in results])

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
            return self._mark(
                lessons(
                    store,
                    self._embedder,
                    query,
                    k=k,
                    type=type,
                    scope=scope,
                    include_retired=include_retired,
                    reranker=self._reranker,
                )
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

    # -- parked items (plan/qmx-parked.md) -------------------------------------------------------

    def _parked(self) -> ParkedRepo:
        return ParkedRepo(self._settings.parked_root)

    def _parked_dict(self, item: ParkedItem) -> dict:
        return item.to_dict(self._settings.parked_stale_days)

    def park(
        self,
        text: str,
        *,
        why: str | None = None,
        scope: str | None = None,
        cwd: str | None = None,
        session: str | None = None,
    ) -> dict:
        """Set an idea aside. The file is written first (always succeeds); indexing is best-effort.

        ``scope`` is the repo key; when omitted it is resolved from ``cwd`` (omit both = global).
        The result carries ``indexed`` and, when other open items look alike, ``similar``.
        """
        if scope is None and cwd:
            scope = canonical_repo_key(cwd)
        item = self._parked().park(text, why=why, scope=scope, session=session)
        out = self._parked_dict(item)
        with self._store() as store:
            out["indexed"] = try_index(store, self._embedder, item)
            if out["indexed"]:
                similar = self._similar_parked(store, item)
                if similar:
                    out["similar"] = similar
        return out

    def _similar_parked(self, store: Store, item: ParkedItem, limit: int = 3) -> list[dict]:
        query = item.title if not item.why else f"{item.title}\n\n{item.why}"
        try:
            [vec] = self._embedder.embed([query])
        except EmbedBackendError:
            return []
        repo = self._parked()
        out: list[dict] = []
        for hit in store.search_vec(vec, k=limit + 1, kind=PARKED_KIND):
            other_id = item_id_from_path(hit.path)
            if other_id in (None, item.id) or hit.distance > PARKED_SIMILAR_MAX_DISTANCE:
                continue
            try:
                other = repo.get(other_id)
            except LookupError:
                continue
            if other.is_open and (other.scope is None or other.scope == item.scope):
                distance = round(hit.distance, 4)
                out.append({"id": other.id, "title": other.title, "distance": distance})
        return out[:limit]

    def list_parked(
        self,
        *,
        scope: str | None = None,
        cwd: str | None = None,
        all_scopes: bool = False,
        status: str = "open",
        query: str | None = None,
        stale_only: bool = False,
        k: int = 20,
    ) -> list[dict]:
        """List parked items — for ``scope`` (or ``cwd``'s repo) plus global, newest first.

        With neither ``scope`` nor ``cwd`` every scope is listed. ``status`` is ``open`` (default) |
        ``done`` | ``dropped`` | ``closed`` | ``all``. ``query`` ranks open items by meaning (closed
        ones, which are not indexed, by keyword); ``stale_only`` keeps the ones due a review.
        """
        if status not in PARKED_LIST_STATUSES:
            raise ValueError(f"status must be one of {PARKED_LIST_STATUSES}, got {status!r}")
        if scope is None and cwd:
            scope = canonical_repo_key(cwd)
        items = self._parked().items(include_archived=status != "open")
        if status == "closed":
            items = [i for i in items if not i.is_open]
        elif status != "all":
            items = [i for i in items if i.status == status]
        items = filter_scope(items, scope, all_scopes=all_scopes or (scope is None and not cwd))
        if stale_only:
            items = [i for i in items if i.is_stale(self._settings.parked_stale_days)]
        if not query:
            return [self._parked_dict(i) for i in sorted(items, key=lambda i: -i.id)[:k]]
        return self._mark(self._rank_parked(items, query, k))

    def _rank_parked(self, items: list[ParkedItem], query: str, k: int) -> list[dict]:
        by_id = {i.id: i for i in items}
        ranked: list[dict] = []
        if any(i.is_open for i in items):
            with self._store() as store:
                hits = search(store, self._embedder, query, k=max(4 * k, 20), kind=PARKED_KIND)
            for hit in hits:
                item = by_id.pop(item_id_from_path(hit.hit.path), None)
                if item is not None:
                    ranked.append({**self._parked_dict(item), "score": round(hit.score, 6)})
        tokens = {t for t in query.lower().split() if len(t) > 2}
        for item in sorted(by_id.values(), key=lambda i: -i.id):  # unindexed: keyword fallback
            haystack = " ".join(filter(None, (item.title, item.why, item.note))).lower()
            if tokens and any(t in haystack for t in tokens):
                ranked.append(self._parked_dict(item))
        return ranked[:k]

    def unpark(self, item_id: int, *, status: str = "done", note: str | None = None) -> dict:
        """Close an item (``done`` | ``dropped``): archived with its note, never deleted."""
        item = self._parked().close(item_id, status=status, note=note)
        with self._store() as store:
            unindex_item(store, item.id)
        return self._parked_dict(item)

    def reopen_parked(self, item_id: int) -> dict:
        """Undo a close: the item is open (and searchable) again."""
        item = self._parked().reopen(item_id)
        out = self._parked_dict(item)
        with self._store() as store:
            out["indexed"] = try_index(store, self._embedder, item)
        return out

    def keep_parked(self, item_id: int) -> dict:
        """Stale review — still relevant: refresh ``updated`` so it is no longer flagged."""
        return self._parked_dict(self._parked().keep(item_id))

    def resume_parked(self, item_id: int, k: int = 3) -> dict:
        """The item plus the chat turns that best match it — the context it was parked from."""
        item = self._parked().get(item_id)
        query = item.title if not item.why else f"{item.title}\n\n{item.why}"
        return {"item": self._parked_dict(item), "context": self.recall(query, k=k)}

    def sync_parked(self) -> dict:
        """Rebuild the parked shadow index from the files (needs the embedding backend)."""
        with self._store() as store:
            stats = sync_index(store, self._embedder, self._parked())
        return {"indexed": stats.indexed, "unchanged": stats.unchanged, "removed": stats.removed}

    def get(self, chunk_id: int) -> dict | None:
        """Full text + location for one chunk, or ``None`` if it is gone/tombstoned."""
        with self._store() as store:
            hit = store.get_chunk(chunk_id)
        return None if hit is None else _hit_dict(hit, score=None, full=True)

    def status(self) -> dict:
        """Index stats + backend health, for ops and the MCP ``status`` tool.

        ``ollama_ok`` is a live ping; ``degraded`` is whether search is currently answering
        BM25-only because the embedding backend failed within the circuit-breaker cooldown.
        """
        with self._store() as store:
            index = store.index_stats()
        return {
            "index": index,
            "embed_model": self._settings.embed_model,
            "ollama_url": self._settings.ollama_url,
            "ollama_ok": ping_ollama(self._settings),
            "degraded": self._embedder.degraded,
        }


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


def _meeting_hit_dict(hit: SearchHit, *, score: float | None) -> dict:
    """A meeting hit, plus the call it belongs to: folder ``YYYY-MM-DD_<name>`` → date + file."""
    out = _hit_dict(hit, score=score)
    path = Path(hit.path) if hit.path else None
    folder = path.parent.name if path else ""
    out["meeting"] = folder
    out["date"] = folder[:10] if _MEETING_FOLDER_DATE.match(folder) else None
    out["file"] = path.name if path else None
    return out
