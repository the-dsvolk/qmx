"""Learnings tier (Capability #3) — distilled, reusable lessons over raw chat recall.

This module orchestrates the store + embed layers into the learning lifecycle:

- :func:`add_learning` — insert a lesson and embed its ``statement``+``detail`` as a
  ``kind='learning'`` document (so it rides the existing vector+FTS+rerank retrieval).
- :func:`lessons` — the **pull** path: semantic search over ``kind='learning'`` re-ranked by
  ``relevance × importance × recency`` (not relevance alone), returning lessons with citations.
- :func:`inject_lessons` — the **push** path: query-free, ``scope``-keyed selection ranked by
  ``importance × recency`` (no query exists yet at ``SessionStart``; see ``plan/qmx-learnings.md``).

Extraction and consolidation (the Qwen passes) live in :mod:`qmx.consolidate`; promotion to
curated memory in :mod:`qmx.promote`.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

from qmx.embed import Embedder
from qmx.index import reindex
from qmx.rerank import Reranker
from qmx.search import search
from qmx.store import Chunk, Learning, Store

LEARNING_TYPES = ("decision", "mistake", "howto")
_PATH_PREFIX = "learning:"

# Blend weights for the pull ranking: relevance × importance × recency (tunable).
W_RELEVANCE = 0.5
W_IMPORTANCE = 0.3
W_RECENCY = 0.2
_RECENCY_HALFLIFE_DAYS = 30.0  # a lesson's recency weight halves every ~month


def learning_doc_path(learning_id: int) -> str:
    """Synthetic document path for a learning's embedded chunk (``learning:<id>``)."""
    return f"{_PATH_PREFIX}{learning_id}"


def embed_text(type: str, statement: str, detail: str | None, topic: str | None = None) -> str:
    """The text embedded/indexed for a learning: type-tagged statement + detail (+ topic)."""
    head = f"[{type}] {topic}".strip() if topic else f"[{type}]"
    body = statement if not detail else f"{statement}\n\n{detail}"
    return f"{head}\n{body}"


def add_learning(
    store: Store,
    embedder: Embedder,
    *,
    type: str,
    statement: str,
    topic: str | None = None,
    scope: str | None = None,
    detail: str | None = None,
    importance: float = 0.5,
    source_anchors: list[dict] | str | None = None,
) -> int:
    """Insert a lesson and embed it as a ``kind='learning'`` document. Returns the ``learning_id``.

    ``source_anchors`` may be a list (JSON-encoded here) or a pre-encoded string; it records the
    session/turn citations so a fired lesson is traceable back to where it was learned.
    """
    if type not in LEARNING_TYPES:
        raise ValueError(f"learning type must be one of {LEARNING_TYPES}, got {type!r}")
    anchors = (
        source_anchors
        if source_anchors is None or isinstance(source_anchors, str)
        else json.dumps(source_anchors)
    )
    learning_id = store.insert_learning(
        type=type,
        statement=statement,
        topic=topic,
        scope=scope,
        detail=detail,
        importance=_clamp01(importance),
        source_anchors=anchors,
    )
    doc_id = store.upsert_document(
        kind="learning", path=learning_doc_path(learning_id), repo=scope or "_global"
    )
    # Trailing [#id] marker guarantees a unique chunk_hash per learning, so two lessons with
    # identical statement+detail stay independently retrievable (the content/mentions store dedups
    # identical chunks, which would otherwise collapse them to one). Negligible for embeddings.
    text = f"{embed_text(type, statement, detail, topic)}\n[#{learning_id}]"
    reindex(store, embedder, doc_id, [Chunk(text=text, symbol=type)])
    store.set_learning_doc(learning_id, doc_id)
    return learning_id


def reembed_learning(store: Store, embedder: Embedder, learning_id: int) -> None:
    """Rebuild a learning's embedded chunk from its current row (after an ``update``)."""
    learning = store.get_learning(learning_id)
    if learning is None:
        return
    doc_id = learning.doc_id or store.upsert_document(
        kind="learning", path=learning_doc_path(learning_id), repo=learning.scope or "_global"
    )
    body = embed_text(learning.type, learning.statement, learning.detail, learning.topic)
    chunk = Chunk(text=f"{body}\n[#{learning_id}]", symbol=learning.type)
    reindex(store, embedder, doc_id, [chunk])
    if learning.doc_id is None:
        store.set_learning_doc(learning_id, doc_id)


def lessons(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    k: int = 5,
    type: str | None = None,
    scope: str | None = None,
    include_global: bool = True,
    reranker: Reranker | None = None,
) -> list[dict]:
    """Pull path: semantic ``kind='learning'`` search re-ranked by relevance×importance×recency.

    Superseded lessons are excluded. ``scope`` (with ``include_global``) filters by repo key;
    ``type`` filters decision/mistake/howto. Each returned (and fired) lesson is ``touch``-ed so its
    ``reuse_count`` reflects use (the promotion gate). Fails soft to ``importance×recency`` order if
    the query yields nothing.
    """
    pool = max(4 * k, 20)
    hits = search(store, embedder, query, k=pool, kind="learning", reranker=reranker)
    relevance = {h.hit.doc_id: h.score for h in hits}
    max_rel = max(relevance.values(), default=0.0) or 1.0

    ranked: list[tuple[float, Learning]] = []
    for h in hits:
        learning = store.learning_by_doc_id(h.hit.doc_id)
        if learning is None or not learning.is_live:
            continue
        if type is not None and learning.type != type:
            continue
        if not _scope_ok(learning, scope, include_global):
            continue
        blended = (
            W_RELEVANCE * (relevance[h.hit.doc_id] / max_rel)
            + W_IMPORTANCE * learning.importance
            + W_RECENCY * _recency(learning)
        )
        ranked.append((blended, learning))

    ranked.sort(key=lambda t: t[0], reverse=True)
    top = ranked[:k]
    for _score, learning in top:
        store.touch_learning(learning.learning_id)
    return [learning_to_dict(learning, score=score) for score, learning in top]


def inject_lessons(store: Store, scope: str | None, *, char_budget: int = 10_000) -> list[Learning]:
    """Push path: query-free, ``scope``-keyed lessons (+ global) by importance×recency, budgeted.

    Returns as many live lessons as fit in ``char_budget`` (the SessionStart ``additionalContext``
    cap). No embedding — injection has no query, so relevance is *project identity*, not meaning.
    """
    # Exclude promoted lessons: they live in curated memory now, so injecting them double-surfaces.
    candidates = store.list_learnings(
        scope=scope, include_global=True, live_only=True, exclude_promoted=True
    )
    candidates.sort(key=lambda le: (le.importance, le.updated_at), reverse=True)
    chosen: list[Learning] = []
    used = 0
    for learning in candidates:
        rendered = render_lesson(learning)
        if chosen and used + len(rendered) > char_budget:
            break
        chosen.append(learning)
        used += len(rendered) + 1
    for learning in chosen:
        store.touch_learning(learning.learning_id)
    return chosen


def render_lesson(learning: Learning) -> str:
    """One-line-ish human rendering of a lesson for injection into session context."""
    scope = learning.scope or "global"
    line = f"- [{learning.type}/{scope}] {learning.statement}"
    if learning.detail:
        line += f" — {learning.detail}"
    return line


def learning_to_dict(learning: Learning, *, score: float | None = None) -> dict:
    """JSON-friendly shape for CLI/MCP, including parsed citations."""
    out: dict = {
        "learning_id": learning.learning_id,
        "type": learning.type,
        "topic": learning.topic,
        "scope": learning.scope,
        "statement": learning.statement,
        "detail": learning.detail,
        "importance": round(learning.importance, 4),
        "reuse_count": learning.reuse_count,
        "promoted_to": learning.promoted_to,
        "citations": _parse_anchors(learning.source_anchors),
    }
    if score is not None:
        out["score"] = round(score, 6)
    return out


def _scope_ok(learning: Learning, scope: str | None, include_global: bool) -> bool:
    if scope is None:
        return True
    if learning.scope == scope:
        return True
    return include_global and learning.scope is None


def _recency(learning: Learning) -> float:
    """Exponential recency weight in ``[0, 1]`` from ``updated_at`` (halves every ~month)."""
    ts = learning.updated_at or learning.created_at
    if not ts:
        return 0.5
    try:
        when = datetime.fromisoformat(ts).replace(tzinfo=UTC)
    except ValueError:
        return 0.5
    age_days = max(0.0, (datetime.now(UTC) - when).total_seconds() / 86400.0)
    return math.exp(-age_days / _RECENCY_HALFLIFE_DAYS)


def _parse_anchors(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return value if isinstance(value, list) else [value]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
