"""Consolidation (Phase B/C) — distill raw chat turns into learnings, then dedup/supersede.

Two Qwen passes over a session's un-``consolidated`` turns:

1. **extract** — read the turns, emit candidate lessons (decision / mistake+correction / howto) as
   JSON, dropping chit-chat.
2. **consolidate** — for each candidate, vector-match existing ``kind='learning'`` and let the model
   decide **new / update / supersede** (a corrected lesson replaces the stale one, not a blind
   INSERT). The source turns are then marked ``consolidated`` so a re-run is idempotent.

Both passes go through the :class:`~qmx.chat.ChatModel` seam, so the pipeline is unit-tested with a
deterministic fake — no live model needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from qmx.chat import ChatModel
from qmx.embed import Embedder
from qmx.learnings import LEARNING_TYPES, add_learning, reembed_learning
from qmx.store import Learning, Store

log = logging.getLogger("qmx.consolidate")

_MAX_TURNS = 200  # cap how many turns feed one extract pass (keeps the prompt bounded)
_MATCH_POOL = 5  # existing lessons shown to the supersede judge per candidate

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
    "`new` (genuinely novel), `update` (same lesson, merge/improve the existing one — set "
    "`target_id`), or `supersede` (the candidate corrects/replaces a now-stale existing one — set "
    "`target_id` to the stale lesson). Prefer `update`/`supersede` over creating near-duplicates. "
    "Return JSON {action, target_id, statement, detail, importance}."
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

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["new", "update", "supersede"]},
        "target_id": {"type": ["integer", "null"]},
        "statement": {"type": "string"},
        "detail": {"type": "string"},
        "importance": {"type": "number"},
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
    """Apply one candidate as new / update / supersede against existing lessons."""
    matches = _nearest_learnings(store, embedder, candidate["statement"], scope)
    decision = _decide(chat, candidate, matches) if matches else {"action": "new"}
    action = decision.get("action", "new")
    statement = decision.get("statement") or candidate["statement"]
    detail = decision.get("detail") or candidate.get("detail")
    importance = decision.get("importance", candidate.get("importance", 0.5))
    target_id = decision.get("target_id")

    if action == "update" and _valid_target(target_id, matches):
        store.update_learning(
            target_id,
            statement=statement,
            detail=detail,
            importance=importance,
            source_anchors=json.dumps(source_anchors) if source_anchors else None,
        )
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
    [vec] = embedder.embed([statement])
    hits = store.search_vec(vec, k=_MATCH_POOL, kind="learning")
    out: list[Learning] = []
    for h in hits:
        learning = store.learning_by_doc_id(h.doc_id)
        if learning is None or not learning.is_live:
            continue
        if scope is not None and learning.scope not in (scope, None):
            continue
        out.append(learning)
    return out


def _decide(chat: ChatModel, candidate: dict, matches: list[Learning]) -> dict:
    listing = "\n".join(
        f"- id={m.learning_id} [{m.type}] {m.statement}"
        + (f" ({m.detail})" if m.detail else "")
        for m in matches
    )
    user = (
        f"NEW candidate lesson:\n[{candidate['type']}] {candidate['statement']}"
        + (f"\ndetail: {candidate['detail']}" if candidate.get("detail") else "")
        + f"\n\nEXISTING similar lessons:\n{listing}"
    )
    return chat.complete_json(CONSOLIDATE_SYSTEM, user, schema=CONSOLIDATE_SCHEMA)


def _valid_target(target_id: object, matches: list[Learning]) -> bool:
    return isinstance(target_id, int) and any(m.learning_id == target_id for m in matches)
