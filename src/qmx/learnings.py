"""Learnings tier (Capability #3) — distilled, reusable lessons over raw chat recall.

This module orchestrates the store + embed layers into the learning lifecycle:

- :func:`add_learning` — insert a lesson and embed its ``statement``+``detail`` as a
  ``kind='learning'`` document (so it rides the existing vector+FTS+rerank retrieval).
- :func:`update_learning` — fix a lesson **in place** (statement/detail/type/topic/scope/
  importance), re-embedding only when an embedded field actually changed.
- :func:`deprecate_learning` / :func:`restore_learning` — soft-retire a lesson (optionally pointing
  at its replacement) and undo. Retirement is metadata-only and enforced inside the search arms, so
  a retired lesson vanishes from *every* read path — not just :func:`lessons` — and comes back for
  free with ``include_retired=True``.
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
from qmx.store import KEEP, Chunk, Keep, Learning, Store

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


def update_learning(
    store: Store,
    embedder: Embedder,
    learning_id: int,
    *,
    type: str = KEEP,
    topic: str | None = KEEP,
    scope: str | None = KEEP,
    statement: str = KEEP,
    detail: str | None = KEEP,
    importance: float = KEEP,
) -> Learning | None:
    """Fix a lesson in place; returns the updated row (``None`` if ``learning_id`` is unknown).

    Every field defaults to :data:`~qmx.store.KEEP` (leave alone) and ``None`` clears a nullable one
    (``topic``/``scope``/``detail``), so ``scope=None`` re-scopes a lesson to global.

    Two costs are avoided deliberately:

    - **No embedding call** unless an *embedded* field changed (``type``/``topic``/``statement``/
      ``detail`` — see :func:`embed_text`). Lowering ``importance``, the most common correction, is
      therefore a purely local write that works with the model backend down.
    - ``scope`` also lives on the learning's ``documents.repo`` row (that is what scope-filtered
      retrieval reads), so it is re-upserted rather than left to drift.
    """
    before = store.get_learning(learning_id)
    if before is None:
        return None
    if not isinstance(type, Keep) and type not in LEARNING_TYPES:
        raise ValueError(f"learning type must be one of {LEARNING_TYPES}, got {type!r}")
    if not isinstance(statement, Keep) and not (statement or "").strip():
        raise ValueError("statement cannot be empty")
    if not isinstance(importance, Keep):
        importance = _clamp01(importance)

    store.update_learning(
        learning_id,
        type=type,
        topic=topic,
        scope=scope,
        statement=statement,
        detail=detail,
        importance=importance,
    )
    after = store.get_learning(learning_id)
    if after is None:  # pragma: no cover - the row was just read above
        return None

    if not isinstance(scope, Keep) and after.scope != before.scope and after.doc_id is not None:
        store.upsert_document(
            kind="learning", path=learning_doc_path(learning_id), repo=after.scope or "_global"
        )
    if embed_text(after.type, after.statement, after.detail, after.topic) != embed_text(
        before.type, before.statement, before.detail, before.topic
    ):
        reembed_learning(store, embedder, learning_id)
    return after


def deprecate_learning(
    store: Store,
    learning_id: int,
    *,
    reason: str | None = None,
    superseded_by: int | None = None,
) -> Learning | None:
    """Soft-retire a lesson and hide it from every search path; returns the row (or ``None``).

    Sets ``deprecated_at`` plus an optional ``reason`` and ``superseded_by`` breadcrumb. Hiding is
    enforced once, inside the search arms (:meth:`~qmx.store.Store.retired_learning_docs`), so a
    retired lesson also disappears from raw ``query --kind learning`` and from the consolidation
    judge's candidate pool — not just from :func:`lessons`. Metadata-only: the chunk, mention and
    embedding are untouched, so this needs no model backend and :func:`restore_learning` is free.
    """
    if store.get_learning(learning_id) is None:
        return None
    if superseded_by is not None:
        if superseded_by == learning_id:
            raise ValueError("a learning cannot supersede itself")
        if store.get_learning(superseded_by) is None:
            raise ValueError(f"superseded_by={superseded_by} is not an existing learning")
    store.deprecate_learning(learning_id, reason=reason, superseded_by=superseded_by)
    return store.get_learning(learning_id)


def restore_learning(store: Store, learning_id: int) -> Learning | None:
    """Undo a retirement: clear ``deprecated_at``/``deprecated_reason`` **and** ``superseded_by``,
    making the lesson fully live again. Metadata-only — no re-embedding."""
    if store.get_learning(learning_id) is None:
        return None
    store.restore_learning(learning_id)
    return store.get_learning(learning_id)


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
    include_retired: bool = False,
    reranker: Reranker | None = None,
) -> list[dict]:
    """Pull path: semantic ``kind='learning'`` search re-ranked by relevance×importance×recency.

    Superseded and soft-retired lessons are excluded unless ``include_retired`` is set — the
    "show me what I retired, with the breadcrumb to its replacement" path, where each result carries
    ``deprecated_at``/``deprecated_reason``/``superseded_by``. ``scope`` (with ``include_global``)
    filters by repo key; ``type`` filters decision/mistake/howto. Each returned (and fired) lesson
    is ``touch``-ed so its ``reuse_count`` reflects use (the promotion gate). Fails soft to
    ``importance×recency`` order if the query yields nothing.
    """
    pool = max(4 * k, 20)
    hits = search(
        store,
        embedder,
        query,
        k=pool,
        kind="learning",
        reranker=reranker,
        include_retired=include_retired,
    )
    relevance = {h.hit.doc_id: h.score for h in hits}
    max_rel = max(relevance.values(), default=0.0) or 1.0

    ranked: list[tuple[float, Learning]] = []
    for h in hits:
        learning = store.learning_by_doc_id(h.hit.doc_id)
        if learning is None or not (learning.is_live or include_retired):
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
        # Retired lessons are surfaced for inspection, not use — don't credit them on the
        # ``reuse_count`` promotion gate.
        if learning.is_live:
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
    # Only present on retired lessons, so a normal result stays as compact as before.
    if learning.is_deprecated:
        out["deprecated_at"] = learning.deprecated_at
        out["deprecated_reason"] = learning.deprecated_reason
    if learning.superseded_by is not None:
        out["superseded_by"] = learning.superseded_by
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


def normalize_importance(value: object) -> float:
    """Coerce a model-supplied ``importance`` onto the documented 0..1 scale.

    The judge is asked for a decimal in 0..1 but sometimes answers on a 1–5 or 1–10 rating scale,
    and ``importance`` is a *ranking weight* (:data:`W_IMPORTANCE` × importance): a stray ``9.0``
    contributes 2.7 while the whole relevance term is capped at :data:`W_RELEVANCE` = 0.5, so one
    mis-scaled lesson outranks every genuinely relevant one, and sails through the promotion gate.

    Rescales by the apparent scale rather than clamping — clamping would flatten every over-range
    lesson to top priority, which is the same bug with a smaller maximum. Unparseable input falls
    back to the 0.5 default rather than raising, since this sits on a model-output path.
    """
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if x != x:  # NaN
        return 0.5
    if x > 5.0:  # looks like a 1–10 rating
        x /= 10.0
    elif x > 1.0:  # looks like a 1–5 rating
        x /= 5.0
    return _clamp01(x)
