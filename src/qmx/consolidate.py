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
from qmx.learnings import (
    LEARNING_TYPES,
    add_learning,
    deprecate_learning,
    normalize_importance,
    reembed_learning,
)
from qmx.store import Learning, Store

log = logging.getLogger("qmx.consolidate")

_MAX_TURNS = 200  # max turns per extract pass (a long session takes several passes)
# Max transcript characters per extract pass (~30K tokens): fits the judge's 64K window with room
# for the reply/thinking, and keeps each pass focused enough for the model to read it all.
_MAX_PASS_CHARS = 100_000
_MATCH_POOL = 5  # live lessons shown to the supersede judge per candidate
# Retired lessons are shown too — otherwise the judge cannot tell a genuinely new lesson from one
# that was deliberately retired, and would re-add it as `new` on the next session (see `drop`).
# Budgeted separately so they can never crowd the live matches out of the prompt.
_MATCH_RETIRED = 3

EXTRACT_SYSTEM = (
    "You distill durable, reusable engineering lessons from a coding session transcript. "
    "Keep only lessons worth recalling next time: a decision and why, a mistake and its "
    "correction, or a repeatable how-to. Drop chit-chat, one-offs, and restated context. "
    "A one-off lookup, status check or Q&A with no reusable insight has no lessons: return an "
    "empty list. Each lesson must generalize beyond this session — no scenario-specific numbers, "
    "dates, deadlines, balances or task status, and no first-person narrative. State only what "
    "the transcript supports and keep its hedges (never turn \"generally\" or \"usually\" into "
    "\"always\"). "
    "Each lesson: a crisp one-sentence `statement`, a `detail` (the why/correction/better way), "
    "a `type` (decision|mistake|howto), a short `topic` slug, and an `importance` as a decimal "
    "between 0.0 and 1.0 (e.g. 0.8 — NOT a 1-5 or 1-10 rating). "
    "Return JSON {\"learnings\": [...]}; return an empty list if nothing is durable. "
    "Before listing lessons, think it through in the `reasoning` field: walk the session, name "
    "each candidate lesson and say why it is or is not durable. Then fill `learnings`."
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
    "Return JSON {action, target_id, statement, detail, importance, reason}. "
    "Before deciding, think it through in the `reasoning` field: compare the candidate with each "
    "existing lesson (same lesson? corrects it? unrelated?). Then fill the remaining fields. "
    "`statement` and `detail` must contain only the lesson itself — never commentary about other "
    "lessons, their ids, or why you chose the action (that belongs in `reasoning`)."
)

# ``reasoning`` is the FIRST property of both schemas on purpose: Ollama turns the schema into a
# grammar that emits fields in ``properties`` order, so the model writes its analysis before it
# commits to lessons / an action — most of thinking mode's benefit without its cost. qmx ignores
# the field's content (it is only logged). Measured 2026-11-02 on 12 real sessions, blind-graded:
# precision 3.8 vs 3.3/5, 7 of 11 sessions won, +46% judge time, no call over 60 s; Ollama thinking
# mode on the same model was ~10x slower.
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
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
    "required": ["reasoning", "learnings"],
}

# Merge guardrail. `update` is the one destructive consolidation edit (it rewrites a lesson in
# place; supersede/deprecate keep the old row), and the judge sometimes "merges" a candidate into
# an unrelated neighbour, replacing that lesson's point. Before applying an update, a second call
# checks that the merged text still states the target's core claim. Embedding similarity can't make
# this call — clobbers usually stay in the same domain (a wash-sale rule rewritten as a lot-schema
# rule).
# Validated 2026-11-02 on 30 real merges: caught all 7 the blind graders flagged as clobbers.
PRESERVE_SYSTEM = (
    "You audit one edit to a knowledge base of engineering lessons. OLD is an existing lesson; "
    "NEW is the text proposed to REPLACE it; CANDIDATE is the newly learned lesson that motivated "
    "the edit. First state OLD's core claim (its main rule, fact or correction) in one sentence. "
    "Then copy, verbatim, the shortest span of NEW that states that same claim — leave it empty "
    "if NEW does not state it. List every fact in NEW that appears in neither OLD nor CANDIDATE. "
    "NEW preserves OLD only if it still states OLD's core claim and is not really about a "
    "different subject. Think it through in `reasoning` first, then answer."
)

PRESERVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "old_core_claim": {"type": "string"},
        "evidence_in_new": {"type": "string"},
        "unsupported_additions": {"type": "array", "items": {"type": "string"}},
        "preserves": {"type": "boolean"},
        "lost": {"type": "string"},
    },
    "required": [
        "reasoning", "old_core_claim", "evidence_in_new", "unsupported_additions", "preserves",
        "lost",
    ],
}

CONSOLIDATE_ACTIONS = ("new", "update", "supersede", "deprecate", "drop")

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": list(CONSOLIDATE_ACTIONS)},
        "target_id": {"type": ["integer", "null"]},
        "statement": {"type": "string"},
        "detail": {"type": "string"},
        "importance": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["reasoning", "action"],
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
    # `update`s refused by the merge guardrail (the merge would lose the target's core claim); each
    # such candidate is stored as a new lesson instead, so it is also counted in ``created``.
    update_blocked: int = 0
    passes: int = 0  # extract passes run (a long session is split into several)
    skipped_small: bool = False  # below the one-off gate; turns left un-consolidated to accumulate
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
    log.debug(
        "judge %s (target %s) for %r: %s",
        action,
        decision.get("target_id"),
        candidate["statement"][:80],
        decision.get("reasoning", ""),
    )
    statement = decision.get("statement") or candidate["statement"]
    detail = decision.get("detail") or candidate.get("detail")
    # Rescale here, at the model boundary: the judge is asked for 0..1 but sometimes answers on a
    # 1–5/1–10 scale, and importance is a ranking weight — an unscaled 9.0 buries every relevant
    # lesson. `add_learning` clamped, but the `update` path wrote the raw number straight through.
    importance = normalize_importance(decision.get("importance", candidate.get("importance", 0.5)))
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
        target = next(m for m in matches if m.learning_id == target_id)
        keeps, lost = _merge_preserves(chat, target, statement, detail, candidate)
        if keeps:
            # Only pass what the merge actually produced: in `update_learning` a ``None`` *clears*
            # the column, so an absent detail/anchors must be omitted rather than sent as None.
            patch: dict = {"statement": statement, "importance": importance}
            if detail is not None:
                patch["detail"] = detail
            if source_anchors:
                patch["source_anchors"] = json.dumps(source_anchors)
            store.update_learning(target_id, **patch, revision_reason="consolidate:update")
            reembed_learning(store, embedder, target_id)
            result.updated += 1
            result.learning_ids.append(target_id)
            return
        # Refused: keep the target intact and store the candidate in its *own* words (the judge's
        # merged text is a blend of target + candidate and would duplicate the target).
        log.info(
            "blocked update of learning #%s (merge would lose: %s); storing candidate as new",
            target_id,
            lost or "its core claim",
        )
        result.update_blocked += 1
        statement = candidate["statement"]
        detail = candidate.get("detail")
        importance = normalize_importance(candidate.get("importance", 0.5))
        action = "new"

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
    min_turns: int = 0,
    min_chars: int = 0,
) -> ConsolidateResult:
    """Distil one chat document's un-consolidated turns into learnings; idempotent on re-run.

    Long sessions are read in several passes (``_MAX_TURNS`` turns / ``_MAX_PASS_CHARS``
    characters each), and each pass's turns are marked consolidated only once that pass is done —
    previously only the first 200 turns were read while *all* were marked, so the rest of a long
    session was silently never distilled. ``min_turns``/``min_chars`` (the CLI passes the
    ``consolidate_min_*`` settings) skip sessions too small to hold a durable lesson, without
    marking them, so a resumed session's turns accumulate until they clear the gate.
    """
    result = ConsolidateResult()
    chunks = store.unconsolidated_chat_chunks(doc_id)
    if not chunks:
        return result
    if len(chunks) < min_turns or sum(len(h.text) for h in chunks) < min_chars:
        log.info("skipping doc %s: %d turns below the one-off gate", doc_id, len(chunks))
        result.skipped_small = True
        return result
    path = chunks[0].path
    for window in _windows(chunks):
        turns = [{"role": h.symbol or "?", "text": h.text, "line": h.start_line} for h in window]
        result.turns_read += len(turns)
        result.passes += 1
        candidates = extract_learnings(chat, turns)
        result.candidates += len(candidates)
        for cand in candidates:
            anchors = [{"transcript_path": path, "line": t["line"]} for t in turns[:3]]
            consolidate_candidate(
                store, embedder, chat, cand, scope=scope, source_anchors=anchors, result=result
            )
        store.mark_consolidated(h.chunk_id for h in window)
        result.chunks_consolidated += len(window)
    return result


def _windows(chunks: list) -> list[list]:
    """Split turns into extract passes of at most ``_MAX_TURNS`` turns / ``_MAX_PASS_CHARS``."""
    windows: list[list] = []
    window: list = []
    chars = 0
    for h in chunks:
        n = len(h.text)
        if window and (len(window) >= _MAX_TURNS or chars + n > _MAX_PASS_CHARS):
            windows.append(window)
            window, chars = [], 0
        window.append(h)
        chars += n
    if window:
        windows.append(window)
    return windows


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


def _merge_preserves(
    chat: ChatModel, target: Learning, statement: str, detail: str | None, candidate: dict
) -> tuple[bool, str]:
    """Ask the judge whether the merged text still states ``target``'s core claim.

    Strict: the judge must quote where the old claim survives, and the quote is checked against
    the merged text in code (a bare "preserves: true" is not enough — a lenient judge approved
    every merge it was shown). Facts in the merge that come from neither the old lesson nor the
    candidate also block it, so a merge cannot smuggle in claims nobody learned. Returns
    ``(preserves, why_not)``. Fails safe: an unusable reply counts as *not* preserved.
    """

    def text(s: str, d: str | None) -> str:
        return s + (f"\ndetail: {d}" if d else "")

    new_text = text(statement, detail)
    reply = chat.complete_json(
        PRESERVE_SYSTEM,
        f"OLD:\n{text(target.statement, target.detail)}\n\nNEW:\n{new_text}"
        f"\n\nCANDIDATE:\n{text(candidate['statement'], candidate.get('detail'))}",
        schema=PRESERVE_SCHEMA,
    )
    evidence = str(reply.get("evidence_in_new") or "").strip()
    quoted = bool(evidence) and _norm(evidence) in _norm(new_text)
    additions = [str(a) for a in reply.get("unsupported_additions") or [] if str(a).strip()]
    if reply.get("preserves") is not True:
        return False, str(reply.get("lost") or "its core claim")
    if not quoted:
        return False, "no verbatim evidence that the merged text still states the old claim"
    if additions:
        return False, "adds facts found in neither lesson: " + "; ".join(additions[:3])
    return True, ""


def _norm(s: str) -> str:
    """Case/whitespace/quote-insensitive form for the verbatim-evidence check."""
    return " ".join(s.lower().replace("`", "").replace('"', "").replace("'", "").split())


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
