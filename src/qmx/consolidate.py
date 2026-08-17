"""Consolidation (Phase B/C) — distill raw chat turns into learnings, then dedup/supersede.

Two Qwen passes over a session's un-``consolidated`` turns:

1. **extract** — read the turns, emit candidate lessons (decision / mistake+correction / howto) as
   JSON, dropping chit-chat.
2. **consolidate** — for each candidate, vector-match existing ``kind='learning'`` and let the model
   decide **new / update / supersede / deprecate / drop** (a corrected lesson replaces the stale
   one, not a blind INSERT). The turns are then marked ``consolidated`` so a re-run is idempotent.

The last two actions close the retirement loop. ``deprecate`` retires a lesson the session proved
wrong when there is no replacement worth storing — the case ``supersede`` cannot express, since it
needs a replacement row. ``drop`` discards a candidate that merely re-learns an already-retired
lesson: retired lessons are hidden from search, so unless they are shown to the judge
(``include_retired=True`` in :func:`_nearest_learnings`) it sees no match and re-adds the lesson as
``new``, quietly undoing the retirement one session later.

Both passes go through the :class:`~qmx.chat.ChatModel` seam, so the pipeline is unit-tested with a
deterministic fake — no live model needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from qmx.chat import ChatModel
from qmx.embed import Embedder
from qmx.learnings import LEARNING_TYPES, add_learning, deprecate_learning, reembed_learning
from qmx.store import Learning, Store

log = logging.getLogger("qmx.consolidate")

_MAX_TURNS = 200  # cap how many turns feed one extract pass (keeps the prompt bounded)
_MATCH_POOL = 5  # live lessons shown to the supersede judge per candidate
# Retired lessons are shown too — otherwise the judge cannot tell a genuinely new lesson from one
# that was deliberately retired, and would re-add it as `new` on the next session (see `drop`).
# Budgeted separately so they can never crowd the live matches out of the prompt.
_MATCH_RETIRED = 3

EXTRACT_SYSTEM = (
    "You distill durable, reusable engineering lessons from a coding session transcript. "
    "Keep only lessons worth recalling next time: a decision and why, a mistake and its "
    "correction, or a repeatable how-to. Drop chit-chat, one-offs, and restated context. "
    "Each lesson: a crisp one-sentence `statement`, a `detail` (the why/correction/better way), "
    "a `type` (decision|mistake|howto), a short `topic` slug, and an `importance` 0..1. "
    "Return JSON {\"learnings\": [...]}; return an empty list if nothing is durable."
)

CONSOLIDATE_SYSTEM = (
    "You maintain a deduplicated store of engineering lessons. Given a NEW candidate lesson and "
    "existing lessons most similar to it, decide one action: "
    "`new` (genuinely novel); "
    "`update` (same lesson, merge/improve the existing one — set `target_id`); "
    "`supersede` (the candidate corrects/replaces a now-stale existing one — set `target_id` to "
    "the stale lesson; the candidate is stored and the stale one retired); "
    "`deprecate` (an existing lesson is simply wrong or obsolete and the candidate is NOT a "
    "replacement worth storing — set `target_id` and a short `reason`; nothing is stored); or "
    "`drop` (the candidate only re-learns a lesson already marked RETIRED — do not resurrect it). "
    "Prefer `update`/`supersede` over creating near-duplicates. Lessons marked RETIRED were "
    "retired on purpose: never `update`, `supersede` or `deprecate` them. "
    "Return JSON {action, target_id, statement, detail, importance, reason}."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "learnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(LEARNING_TYPES)},
                    "topic": {"type": "string"},
                    "statement": {"type": "string"},
                    "detail": {"type": "string"},
                    "importance": {"type": "number"},
                },
                "required": ["type", "statement"],
            },
        }
    },
    "required": ["learnings"],
}

CONSOLIDATE_ACTIONS = ("new", "update", "supersede", "deprecate", "drop")

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(CONSOLIDATE_ACTIONS)},
        "target_id": {"type": ["integer", "null"]},
        "statement": {"type": "string"},
        "detail": {"type": "string"},
        "importance": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass(slots=True)
class ConsolidateResult:
    turns_read: int = 0
    candidates: int = 0
    created: int = 0
    updated: int = 0
    superseded: int = 0
    deprecated: int = 0  # existing lessons retired by the judge (no replacement stored)
    dropped: int = 0  # candidates discarded as re-learning an already-retired lesson
    chunks_consolidated: int = 0
    learning_ids: list[int] = field(default_factory=list)


def extract_learnings(chat: ChatModel, turns: list[dict]) -> list[dict]:
    """Run the extract pass over ``turns`` (``[{role, text, line}]``); return candidate lessons."""
    if not turns:
        return []
    convo = "\n\n".join(f"{t.get('role', '?').upper()}: {t.get('text', '')}" for t in turns)
    reply = chat.complete_json(EXTRACT_SYSTEM, convo, schema=EXTRACT_SCHEMA)
    out: list[dict] = []
    for c in reply.get("learnings", []):
        if isinstance(c, dict) and c.get("type") in LEARNING_TYPES and c.get("statement"):
            out.append(c)
    return out


def consolidate_candidate(
    store: Store,
    embedder: Embedder,
    chat: ChatModel,
    candidate: dict,
    *,
    scope: str | None,
    source_anchors: list[dict] | None,
    result: ConsolidateResult,
) -> None:
    """Apply one candidate as new / update / supersede / deprecate / drop against existing lessons.

    ``deprecate`` and ``drop`` are the retirement half of the loop: the first retires an existing
    lesson the session proved wrong (storing no replacement), the second discards a candidate that
    merely re-learns an already-retired lesson, so retirement is not silently undone next session.
    Both are counted on the result rather than logged away, since a candidate that vanishes without
    a trace is indistinguishable from one that was never extracted.
    """
    matches = _nearest_learnings(store, embedder, candidate["statement"], scope)
    decision = _decide(chat, candidate, matches) if matches else {"action": "new"}
    action = decision.get("action", "new")
    statement = decision.get("statement") or candidate["statement"]
    detail = decision.get("detail") or candidate.get("detail")
    importance = decision.get("importance", candidate.get("importance", 0.5))
    target_id = decision.get("target_id")

    if action == "drop":
        retired = [m.learning_id for m in matches if not m.is_live]
        log.info(
            "dropped candidate %r: re-learns retired lesson(s) %s",
            candidate["statement"][:80],
            retired or "(none matched)",
        )
        result.dropped += 1
        return

    if action == "deprecate" and _valid_target(target_id, matches):
        reason = decision.get("reason") or f"contradicted while learning: {statement}"
        deprecate_learning(store, target_id, reason=reason)
        log.info("deprecated learning #%s: %s", target_id, reason)
        result.deprecated += 1
        return

    if action == "update" and _valid_target(target_id, matches):
        # Only pass what the merge actually produced: in `update_learning` a ``None`` *clears* the
        # column, so an absent detail/anchors must be omitted rather than sent as None.
        patch: dict = {"statement": statement, "importance": importance}
        if detail is not None:
            patch["detail"] = detail
        if source_anchors:
            patch["source_anchors"] = json.dumps(source_anchors)
        store.update_learning(target_id, **patch)
        reembed_learning(store, embedder, target_id)
        result.updated += 1
        result.learning_ids.append(target_id)
        return

    new_id = add_learning(
        store,
        embedder,
        type=candidate["type"],
        statement=statement,
        topic=candidate.get("topic"),
        scope=scope,
        detail=detail,
        importance=importance,
        source_anchors=source_anchors,
    )
    result.learning_ids.append(new_id)
    if action == "supersede" and _valid_target(target_id, matches):
        store.supersede_learning(target_id, new_id)
        result.superseded += 1
    else:
        result.created += 1


def consolidate_session(
    store: Store,
    embedder: Embedder,
    chat: ChatModel,
    doc_id: int,
    *,
    scope: str | None = None,
) -> ConsolidateResult:
    """Distil one chat document's un-consolidated turns into learnings; idempotent on re-run."""
    result = ConsolidateResult()
    chunks = store.unconsolidated_chat_chunks(doc_id)
    if not chunks:
        return result
    path = chunks[0].path
    turns = [
        {"role": h.symbol or "?", "text": h.text, "line": h.start_line}
        for h in chunks[:_MAX_TURNS]
    ]
    result.turns_read = len(turns)
    candidates = extract_learnings(chat, turns)
    result.candidates = len(candidates)
    for cand in candidates:
        anchors = [{"transcript_path": path, "line": t["line"]} for t in turns[:3]]
        consolidate_candidate(
            store, embedder, chat, cand, scope=scope, source_anchors=anchors, result=result
        )
    store.mark_consolidated(h.chunk_id for h in chunks)
    result.chunks_consolidated = len(chunks)
    return result


def _nearest_learnings(
    store: Store, embedder: Embedder, statement: str, scope: str | None
) -> list[Learning]:
    """Nearest ``kind='learning'`` matches: up to ``_MATCH_POOL`` live + ``_MATCH_RETIRED`` retired.

    Retired lessons are fetched with ``include_retired=True`` *and* budgeted separately, so showing
    them to the judge can never displace a live match it might have merged into.
    """
    [vec] = embedder.embed([statement])
    over_fetch = (_MATCH_POOL + _MATCH_RETIRED) * 2  # kind/scope filtering trims the pool
    hits = store.search_vec(vec, k=over_fetch, kind="learning", include_retired=True)
    live: list[Learning] = []
    retired: list[Learning] = []
    for h in hits:
        learning = store.learning_by_doc_id(h.doc_id)
        if learning is None:
            continue
        if scope is not None and learning.scope not in (scope, None):
            continue
        if learning.is_live:
            if len(live) < _MATCH_POOL:
                live.append(learning)
        elif len(retired) < _MATCH_RETIRED:
            retired.append(learning)
    return live + retired


def _decide(chat: ChatModel, candidate: dict, matches: list[Learning]) -> dict:
    listing = "\n".join(_describe_match(m) for m in matches)
    user = (
        f"NEW candidate lesson:\n[{candidate['type']}] {candidate['statement']}"
        + (f"\ndetail: {candidate['detail']}" if candidate.get("detail") else "")
        + f"\n\nEXISTING similar lessons:\n{listing}"
    )
    return chat.complete_json(CONSOLIDATE_SYSTEM, user, schema=CONSOLIDATE_SCHEMA)


def _describe_match(m: Learning) -> str:
    """One prompt line for a candidate's neighbour, flagged ``[RETIRED: why]`` when not live."""
    line = f"- id={m.learning_id} [{m.type}] {m.statement}"
    if m.detail:
        line += f" ({m.detail})"
    if not m.is_live:
        why = m.deprecated_reason or (
            f"superseded by #{m.superseded_by}" if m.superseded_by else "retired"
        )
        line += f"  [RETIRED: {why}]"
    return line


def _valid_target(target_id: object, matches: list[Learning]) -> bool:
    """A target is actionable only if the judge was actually shown it **and** it is still live.

    Retired lessons appear in the prompt for context; editing or re-retiring one would undo a
    deliberate retirement, so they are never valid targets. An invalid target falls through to the
    insert path — a contradictory decision costs a possible duplicate, never a lost candidate.
    """
    return isinstance(target_id, int) and any(
        m.learning_id == target_id and m.is_live for m in matches
    )
