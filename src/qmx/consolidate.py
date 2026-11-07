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

Both actions that retire a live lesson — ``supersede`` and ``deprecate`` — must first pass the
retirement guardrail (:func:`_contradiction_verified`): the judge quotes the span of the candidate
that contradicts the old lesson, and code checks the quote is really there. A refused ``deprecate``
stores the candidate as new; a refused ``supersede`` sends it through the insert-time near-duplicate
check, since a candidate that does not contradict its target usually restates or extends it.

Both passes go through the :class:`~qmx.chat.ChatModel` seam, so the pipeline is unit-tested with a
deterministic fake — no live model needed.

**Filing rules.** Which prompts the judge gets, and how strictly code checks its edits, is a
:class:`JudgeConfig` (``consolidate_session(..., judge=judge_config(settings))``; ``None`` = the
module constants below, i.e. today's behaviour). ``judge_filing_rules = "current"`` (default) is the
validated production set. ``"v2"`` compares MAIN RULES instead of topics, where a lesson's main rule
is the first sentence of its statement, computed in code (:func:`main_rule`) and handed to every
check; adds a deterministic headline backstop (:func:`headline_kept`: a merged statement must begin
with the old first sentence, extensions and corrected numbers allowed), an off-rule-additions check
in the merge check (logged by default; with ``judge_offrule_block`` the merge is refused and the
candidate stored as new, never dropped) and a two-quote retirement check (the contradicting words of
the candidate *and* at least three words of the old main rule that become false, both verified in
code). ``judge_reasoning_field`` picks the per-model wrapper around one
shared rule text: a reasoning-first JSON field (Qwen3.8, thinking off) or none (a model with native
thinking, e.g. Nemotron 3.5 Super).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from qmx.chat import ChatModel, ChatReplyError
from qmx.config import Settings
from qmx.embed import Embedder
from qmx.learnings import (
    LEARNING_TYPES,
    add_learning,
    deprecate_learning,
    normalize_importance,
    reembed_learning,
)
from qmx.lesson_text import (  # noqa: F401  (re-exported: eval harness and tests use C.*)
    _ABBREV,
    _FOLD,
    _SENT_END,
    _TYPE_TAG,
    _fold,
    main_rule,
    one_sentence,
    strip_type_tags,
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

# Output cap for the short judge calls (decision, merge check, retirement check, dedup check). Their
# replies peaked at ~2,100 tokens on 57 replayed real calls (median ~400); 3072 leaves headroom and
# bounds a runaway reply to ~2-3 minutes instead of the 8192-token extract cap. The default of the
# ``chat_short_max_tokens`` setting (a thinking model needs more: thinking shares the cap), and
# what a call without a ``judge`` uses.
SHORT_REPLY_MAX_TOKENS = 3072

# Appended to every judge prompt. The judge often quotes inside its free-text fields (`reasoning`
# especially); an unescaped double quote closes the JSON string early. With the fixed-layout grammar
# (qmx.grammar) that only truncates the field, but it still cuts the reasoning short.
_QUOTE_RULE = (
    " In every JSON text value, quote with single quotes ('like this'), never double quotes: an "
    "unescaped double quote ends the value early."
)

EXTRACT_SYSTEM = (
    "You distill durable, reusable engineering lessons from a coding session transcript. "
    "Keep only lessons worth recalling next time: a decision and why, a mistake and its "
    "correction, or a repeatable how-to. Drop chit-chat, one-offs, and restated context. "
    "A one-off lookup, status check or Q&A with no reusable insight has no lessons: return an "
    "empty list. Write each lesson so it is reusable beyond this session: drop scenario-specific "
    "values (balances, dates, deadlines, one-off task status) and first-person narrative, but "
    "keep the concrete commands, settings, thresholds and names that make it actionable. State "
    "only what "
    "the transcript supports and keep its hedges (never turn \"generally\" or \"usually\" into "
    "\"always\"). "
    "Each lesson: a crisp one-sentence `statement`, a `detail` (the why/correction/better way), "
    "a `type` (decision|mistake|howto), a short `topic` slug, and an `importance` as a decimal "
    "between 0.0 and 1.0 (e.g. 0.8 — NOT a 1-5 or 1-10 rating). "
    "Return JSON {\"learnings\": [...]}; return an empty list if nothing is durable. "
    "Before listing lessons, think it through in the `reasoning` field: walk the session, name "
    "each candidate lesson and say why it is or is not durable. Then fill `learnings`."
) + _QUOTE_RULE

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
) + _QUOTE_RULE

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
) + _QUOTE_RULE

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

# Retirement guardrail. `deprecate` and `supersede` both hide a live lesson, and judges have retired
# lessons the session supported (#380 three times by deprecate across eval runs; in the 2026-11-04
# end-to-end eval, three valid lessons by supersede, #380 among them). Before retiring, the judge
# must quote the part of the CANDIDATE that contradicts the old lesson; code checks the quote really
# is in the candidate. One prompt serves both actions: a supersede may only retire OLD because the
# candidate makes it stale ("contradicts or obsoletes"); a candidate that adds detail to or narrows
# OLD is an update, not a supersede.
DEPRECATE_SYSTEM = (
    "You review a proposal to RETIRE an existing engineering lesson as wrong or obsolete. OLD is "
    "the lesson; CANDIDATE is what was just learned. Copy, verbatim, the shortest span of "
    "CANDIDATE that contradicts or obsoletes OLD — leave it empty if CANDIDATE does not actually "
    "contradict OLD (adding detail, narrowing it, or agreeing with it is NOT a contradiction). "
    "Think it through in `reasoning` first, then answer."
) + _QUOTE_RULE

DEPRECATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "contradiction_quote": {"type": "string"},
        "contradicts": {"type": "boolean"},
    },
    "required": ["reasoning", "contradiction_quote", "contradicts"],
}

# Insert-time near-duplicate check. Reading whole sessions yields more candidates, and the judge
# often files one as `new` although an existing lesson already states it. Similarity alone
# cannot decide (eval 2026-11-03: true duplicates scored 0.68-0.79, unrelated neighbours up to
# 0.88), so any `new` whose nearest live lesson is >= ``DEDUP_MIN_SIMILARITY`` gets one question.
DEDUP_MIN_SIMILARITY = 0.65
DEDUP_SYSTEM = (
    "You check whether a newly learned lesson is already in a knowledge base. EXISTING is a "
    "stored lesson; CANDIDATE is new. Decide whether CANDIDATE is the same lesson as EXISTING — "
    "the same core rule, fact or correction, possibly reworded, narrower, or a special case of "
    "it. If it is, decide whether CANDIDATE adds a durable, specific fact that EXISTING lacks; if "
    "so, write a merged statement and detail that keep EXISTING's core claim word-for-word where "
    "possible and add the new fact. Think it through in `reasoning` first, then answer."
) + _QUOTE_RULE
DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "same_lesson": {"type": "boolean"},
        "adds_new_fact": {"type": "boolean"},
        "merged_statement": {"type": "string"},
        "merged_detail": {"type": "string"},
    },
    "required": ["reasoning", "same_lesson", "adds_new_fact", "merged_statement", "merged_detail"],
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

# -- Filing rules v2 (judge_filing_rules = "v2") ---------------------------------------------------
# One shared rule text per call with two per-model slots. Evidence and the replay results behind
# each sentence: ~/knowledge/qwen-judge-eval/prompt-research/ (FINDINGS.md, PROPOSAL.md section A).
# The substrings tests/fakes.py routes on ("distill", "audit", "already in a knowledge base",
# "proposal to RETIRE") must stay in these texts.
FILING_RULES = ("current", "v2")
# How the decision call lists the existing lessons (``judge_listing``): "inline" (default, one line
# per lesson, byte-identical to what the Qwen3.8 judge was validated with) or "labelled" (one
# labelled field per line: the code-computed main rule, the full statement when it has more than
# that, the detail). See :func:`_describe_match_labelled`.
LISTINGS = ("inline", "labelled")

QUOTE_RULE_V2 = (
    " In every JSON text value, quote with single quotes ('like this'), never double quotes: an "
    "unescaped double quote ends the value early. Never wrap a whole value in quotes."
)

# Wrapper slots, keyed by ``judge_reasoning_field``: True = the reasoning-first JSON field (Qwen3.8,
# thinking off), False = native thinking (the steps go to the model's own reasoning).
_V2_WORK = {
    True: "Before deciding, work through these steps in the `reasoning` field:",
    False: "Work through these steps before you answer:",
}
_V2_RETURN = {
    True: "Return JSON {reasoning, action, target_id, statement, detail, importance, reason}; "
          "`reasoning` comes first and holds your working, never the lesson text.",
    False: "Return JSON {action, target_id, statement, detail, importance, reason}.",
}
_V2_THINK = {True: " Think it through in `reasoning` first, then answer.", False: ""}

CONSOLIDATE_RULES_V2 = (
    "You maintain a knowledge base of engineering lessons. You get a NEW candidate lesson and the "
    "existing lessons most similar to it. Decide ONE action by comparing MAIN RULES, not topics. A "
    "lesson's main rule is its one-sentence point; for an existing lesson it is the first sentence "
    "of its statement. Two lessons share a main rule when someone who already follows the existing "
    "lesson's main rule would do or know the same thing after reading the candidate's: a reworded, "
    "narrower or more concrete version, an example, or the same advice for one tool or project "
    "counts as the same main rule. A different piece of advice, a different fact or a different "
    "decision is a different main rule, even when it is about the same tool, data, project or "
    "topic. {WORK} (1) Copy the main rule of the closest existing lesson. (2) State the "
    "candidate's main rule in one sentence. (3) Same main rule? (4) Does the candidate make that "
    "existing main rule itself false or obsolete (a reversed decision, a changed tool behaviour, "
    "or a main rule that rested on a wrong fact)? Find the candidate's exact words. A corrected "
    "number, flag, path, price or percentage does NOT make a main rule false, and neither does "
    "correcting something the existing lesson never claimed. (5) Does the candidate add a durable "
    "fact the existing lesson lacks? Then choose: `new`: the candidate's main rule differs from "
    "every live lesson's main rule. `update`: same main rule as the target, and the candidate adds "
    "a fact, example, condition or caveat, or corrects a detail; set `target_id`. `drop`: same "
    "main rule as a live lesson and nothing durable to add, or the candidate only re-learns a "
    "lesson marked RETIRED. `supersede`: the target's main rule is now false and the candidate is "
    "the corrected rule; set `target_id` and put in `reason` the candidate's exact words that make "
    "it false and the words of the old main rule that become false. `deprecate`: the target's main "
    "rule is now false and the candidate is not worth storing; set `target_id` and the same kind "
    "of `reason`. Never `supersede` or `deprecate` because the candidate confirms, extends, "
    "narrows, gives an example of or offers an alternative to the old lesson, or corrects a minor "
    "detail; if you are not sure the old main rule is false, choose `update`. If the main rules "
    "are the same, choose `update` or `drop`, not `new`; choose `new` only when they differ. "
    "Never `update`, `supersede` or `deprecate` a lesson marked RETIRED. Writing an "
    "`update`: your `statement` and `detail` REPLACE the stored lesson. `statement` must begin "
    "with the existing statement's first sentence copied word for word; you may change a number in "
    "it that the candidate corrects and end it with a short clause, but put every other new fact "
    "in `detail`. If the candidate shows that first sentence is wrong, the action is `supersede`, "
    "not `update`. `detail` keeps every existing fact (numbers, commands, paths, names, caveats) "
    "and adds the candidate's new facts about the same main rule in a sentence or two. If the new "
    "facts are about a different rule, the action is `new`. Writing a `new` lesson: a one-sentence "
    "`statement` with the reusable rule and a `detail` with the why and the specifics. In every "
    "text field: no personal names, one-off amounts or dates unless they are the point; no "
    "commentary about other lessons, their ids, similarity scores or why you chose the action; no "
    "[type] tags. {RETURN}"
)

DEDUP_RULES_V2 = (
    "You check whether a newly learned lesson is already in a knowledge base. EXISTING is a stored "
    "lesson; its main rule is the first sentence of its statement. CANDIDATE is new. Compare MAIN "
    "RULES, not topics. CANDIDATE is the same lesson when someone who already follows EXISTING's "
    "main rule would do or know the same thing after reading CANDIDATE's main rule: a reworded, "
    "narrower or more concrete version, an example, or the same advice for one tool or project. A "
    "different piece of advice, fact or decision is a different lesson, even about the same tool, "
    "data, project or topic. If it is the same lesson, decide whether CANDIDATE adds a durable, "
    "specific fact that EXISTING lacks. If it does, write the merged lesson: `merged_statement` "
    "begins with EXISTING's first sentence copied word for word (you may end it with a short "
    "clause), and `merged_detail` keeps every fact of EXISTING's detail and adds only CANDIDATE's "
    "new facts about the same main rule, in a sentence or two. No commentary about other lessons, "
    "ids or this check, and no [type] tags.{THINK}"
)

PRESERVE_RULES_V2 = (
    "You audit one edit to a knowledge base of engineering lessons. OLD is an existing lesson "
    "and OLD MAIN RULE is its first sentence. NEW is the text proposed to REPLACE OLD; "
    "CANDIDATE is the newly learned lesson that motivated the edit. (1) Copy, verbatim, the "
    "shortest span of NEW that states OLD MAIN RULE; leave it empty if NEW does not state it. "
    "(2) List up to three facts of OLD (a number, command, path, name, condition or caveat) "
    "that NEW no longer states; ignore commentary about other lessons or ids. (3) List up to "
    "three sentences or clauses of NEW that state a different rule than OLD MAIN RULE, even "
    "if they are about the same tool, data or topic. (4) List up to three facts in NEW that "
    "appear in neither OLD nor CANDIDATE. NEW preserves OLD only if it still states OLD MAIN "
    "RULE and is still a lesson about that rule.{THINK}"
)

# `old_core_claim` is gone on purpose: code supplies OLD MAIN RULE, so the judge can no longer pick
# a minor sentence as the "core claim" a clobber happened to keep (FINDINGS 1.4, X-4057-med #2005).
# The three lists are capped at 3 items (enforced by the structured-output grammar): they are only
# logged (off-rule blocks only with judge_offrule_block, where one item is enough), and on long
# lessons enumerating every fact ran merge checks into the token cap (eval 2026-11-05).
PRESERVE_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "evidence_in_new": {"type": "string"},
        "lost_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "off_rule_additions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "unsupported_additions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "preserves": {"type": "boolean"},
        "lost": {"type": "string"},
    },
    "required": [
        "reasoning", "evidence_in_new", "lost_facts", "off_rule_additions",
        "unsupported_additions", "preserves", "lost",
    ],
}

DEPRECATE_RULES_V2 = (
    "You review a proposal to RETIRE an existing engineering lesson as wrong or obsolete. OLD is "
    "the lesson and OLD MAIN RULE is its first sentence; CANDIDATE is what was just learned. OLD "
    "may be retired only if CANDIDATE makes OLD MAIN RULE itself false or obsolete: a reversed "
    "decision, a changed tool behaviour, or a main rule that rested on a wrong fact. Copy, "
    "verbatim, the shortest span of CANDIDATE that makes it false into `contradiction_quote`, and "
    "the words of OLD MAIN RULE that become false into `false_in_old`. Leave both empty and set "
    "`contradicts` to false when CANDIDATE adds detail to OLD, narrows or extends it, agrees with "
    "it, offers an alternative, states a different rule about the same topic, corrects something "
    "OLD never claimed (for example an earlier statement in the session), or corrects only a "
    "number, flag, path or price.{THINK}"
)

DEPRECATE_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "contradiction_quote": {"type": "string"},
        "false_in_old": {"type": "string"},
        "contradicts": {"type": "boolean"},
    },
    "required": ["reasoning", "contradiction_quote", "false_in_old", "contradicts"],
}

# Sentences that point the judge at the `reasoning` field in today's ("current") prompts; removed,
# with the schema property, when ``judge_reasoning_field`` is false. Identical to what the eval
# harness stripped for its native-thinking arms (P-S2, X-4057-*), so those runs stay reproducible.
_CURRENT_REASONING_SENTENCES = (
    " Before listing lessons, think it through in the `reasoning` field: walk the session, name "
    "each candidate lesson and say why it is or is not durable. Then fill `learnings`.",
    "Before deciding, think it through in the `reasoning` field: compare the candidate with each "
    "existing lesson (same lesson? corrects it? unrelated?). Then fill the remaining fields. ",
    " (that belongs in `reasoning`)",
    " Think it through in `reasoning` first, then answer.",
)


@dataclass(frozen=True, slots=True)
class JudgePrompts:
    """The system prompt and JSON schema of each of the five judge calls."""

    extract_system: str
    extract_schema: dict
    consolidate_system: str
    consolidate_schema: dict
    preserve_system: str
    preserve_schema: dict
    deprecate_system: str
    deprecate_schema: dict
    dedup_system: str
    dedup_schema: dict


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """How the judge is asked and checked: prompts, short-call cap and filing rules.

    Build one with :func:`judge_config` (from Settings) or :func:`build_judge`. ``None`` wherever a
    ``judge`` is accepted means today's module constants, read at call time — so the eval harness
    can still monkeypatch ``CONSOLIDATE_SYSTEM`` or ``SHORT_REPLY_MAX_TOKENS``.
    """

    prompts: JudgePrompts
    short_max_tokens: int = SHORT_REPLY_MAX_TOKENS
    filing_rules: str = "current"
    reasoning_field: bool = True
    # v2 only: refuse (True) or only log (False) a merge that adds a different rule.
    offrule_block: bool = False
    # How the decision call lists existing lessons: "inline" (default) or "labelled".
    listing: str = "inline"

    @property
    def v2(self) -> bool:
        return self.filing_rules == "v2"


def _without_reasoning(schema: dict) -> dict:
    props = {k: v for k, v in schema["properties"].items() if k != "reasoning"}
    required = [r for r in schema.get("required", []) if r != "reasoning"]
    return {**schema, "properties": props, "required": required}


def _strip_reasoning_sentences(text: str) -> str:
    for sentence in _CURRENT_REASONING_SENTENCES:
        text = text.replace(sentence, "")
    return text


def judge_prompts(filing_rules: str = "current", reasoning_field: bool = True) -> JudgePrompts:
    """The five prompts + schemas for ``filing_rules`` ("current" | "v2") and the model wrapper.

    "current" with the reasoning field returns today's module constants themselves (read now, so a
    monkeypatched constant is honoured). Extraction text is the same under both rule sets; only the
    reasoning wrapper changes it.
    """
    rf = bool(reasoning_field)
    extract_system, extract_schema = EXTRACT_SYSTEM, EXTRACT_SCHEMA
    if not rf:
        extract_system = _strip_reasoning_sentences(extract_system)
        extract_schema = _without_reasoning(extract_schema)
    if filing_rules == "current":
        if rf:
            return JudgePrompts(
                EXTRACT_SYSTEM, EXTRACT_SCHEMA, CONSOLIDATE_SYSTEM, CONSOLIDATE_SCHEMA,
                PRESERVE_SYSTEM, PRESERVE_SCHEMA, DEPRECATE_SYSTEM, DEPRECATE_SCHEMA,
                DEDUP_SYSTEM, DEDUP_SCHEMA,
            )
        strip, schema = _strip_reasoning_sentences, _without_reasoning
        return JudgePrompts(
            extract_system, extract_schema,
            strip(CONSOLIDATE_SYSTEM), schema(CONSOLIDATE_SCHEMA),
            strip(PRESERVE_SYSTEM), schema(PRESERVE_SCHEMA),
            strip(DEPRECATE_SYSTEM), schema(DEPRECATE_SCHEMA),
            strip(DEDUP_SYSTEM), schema(DEDUP_SCHEMA),
        )
    if filing_rules != "v2":
        raise ValueError(f"judge_filing_rules must be one of {FILING_RULES}, got {filing_rules!r}")
    think = _V2_THINK[rf]

    def schema_for(s: dict) -> dict:
        return s if rf else _without_reasoning(s)

    decide = CONSOLIDATE_RULES_V2.replace("{WORK}", _V2_WORK[rf])
    decide = decide.replace("{RETURN}", _V2_RETURN[rf])
    return JudgePrompts(
        extract_system, extract_schema,
        decide + QUOTE_RULE_V2, schema_for(CONSOLIDATE_SCHEMA),
        PRESERVE_RULES_V2.replace("{THINK}", think) + QUOTE_RULE_V2, schema_for(PRESERVE_SCHEMA_V2),
        DEPRECATE_RULES_V2.replace("{THINK}", think) + QUOTE_RULE_V2,
        schema_for(DEPRECATE_SCHEMA_V2),
        DEDUP_RULES_V2.replace("{THINK}", think) + QUOTE_RULE_V2, schema_for(DEDUP_SCHEMA),
    )


def build_judge(
    filing_rules: str = "current",
    *,
    reasoning_field: bool = True,
    short_max_tokens: int | None = None,
    offrule_block: bool = False,
    listing: str = "inline",
) -> JudgeConfig:
    """A :class:`JudgeConfig`; ``short_max_tokens=None`` = ``SHORT_REPLY_MAX_TOKENS`` (read now).

    ``build_judge()`` is exactly today's judge. For Nemotron 3.5 Super:
    ``build_judge("v2", reasoning_field=False, short_max_tokens=16384)``.
    """
    rules = (filing_rules or "current").strip().lower()
    if rules not in FILING_RULES:
        raise ValueError(f"judge_filing_rules must be one of {FILING_RULES}, got {filing_rules!r}")
    cap = SHORT_REPLY_MAX_TOKENS if short_max_tokens is None else int(short_max_tokens)
    shape = (listing or "inline").strip().lower()
    if shape not in LISTINGS:
        raise ValueError(f"judge_listing must be one of {LISTINGS}, got {listing!r}")
    return JudgeConfig(
        prompts=judge_prompts(rules, reasoning_field),
        short_max_tokens=cap,
        filing_rules=rules,
        reasoning_field=bool(reasoning_field),
        offrule_block=bool(offrule_block),
        listing=shape,
    )


def judge_config(settings: Settings) -> JudgeConfig:
    """The judge the ``judge_filing_rules`` / ``judge_reasoning_field`` /
    ``chat_short_max_tokens`` / ``judge_offrule_block`` / ``judge_listing`` settings select
    (defaults: today's judge, byte for byte)."""
    return build_judge(
        settings.judge_filing_rules,
        reasoning_field=settings.judge_reasoning_field,
        short_max_tokens=settings.chat_short_max_tokens,
        offrule_block=settings.judge_offrule_block,
        listing=settings.judge_listing,
    )


def _resolve(judge: JudgeConfig | None) -> JudgeConfig:
    """``None`` -> today's judge from the module constants as they are *now*."""
    return judge if judge is not None else build_judge()


# -- Main rule, normalization, headline backstop (filing rules v2) --------------------------------
# A lesson's main rule is the first sentence of its statement, computed here and used by every v2
# prompt and check: letting the judge choose the "core claim" let clobbers through (FINDINGS 1.4),
# and 22% of live statements have more than one sentence.
# _TYPE_TAG, _ABBREV, _SENT_END, _FOLD, _fold and main_rule live in qmx.lesson_text (shared with
# the one-sentence lesson template applied on every save) and are imported above.
# A number with an optional unit: "$1,200.50", "65%", "88448GB", "3x". A merged statement may
# correct one of these in the old first sentence and still keep its headline.
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|x|k|m|gb|gib|tb|tib|mib))?", re.I)
# An apostrophe inside a word (don't, user's) — not a quote mark.
_APOSTROPHE = re.compile(r"(?<=\w)'(?=\w)")


def _norm_v2(s: str) -> str:
    """:func:`_norm`, plus Unicode folding and a stripped leading ``[type]`` tag (v2 checks)."""
    s = _TYPE_TAG.sub("", _fold(s)).lower()
    return " ".join(s.replace("`", "").replace('"', "").replace("'", "").split())


def headline_kept(old_statement: str, new_statement: str) -> bool:
    """Does NEW's statement still begin with OLD's main rule?

    Matching ignores case, whitespace, quotes, backticks, a ``[type]`` tag and dash style, and must
    end on a word boundary. Allowed: a pure extension (more words after the old first sentence) and
    a version that differs only in numbers (a corrected detail). Anything else — a reworded,
    narrowed or replaced first sentence — is refused before the LLM merge check is even asked. On
    the eval merges it caught every known clobber and fired 0 times under the v2 prompts, which ask
    for the first sentence word for word (FINDINGS 2.4); under today's prompts it would also block
    acceptable rewordings, so it ships only with v2.
    """
    head = _norm_v2(main_rule(old_statement)).rstrip(" .!?;:")
    new = _norm_v2(new_statement or "")
    if not head:
        return True

    def starts(n: str, h: str) -> bool:
        return n.startswith(h) and (len(n) == len(h) or not n[len(h)].isalnum())

    return starts(new, head) or starts(_NUM.sub("#", new), _NUM.sub("#", head))


def _clean_text(value: object) -> object:
    """v2 hygiene on a judge-written statement/detail: drop a leading ``[type]`` tag copied from the
    listing and one pair of single quotes wrapping the whole value (both seen from Super)."""
    if not isinstance(value, str):
        return value
    out = _TYPE_TAG.sub("", value).strip()
    if len(out) >= 2 and out[0] == out[-1] == "'" and _APOSTROPHE.sub("", out).count("'") == 2:
        out = _TYPE_TAG.sub("", out[1:-1]).strip()
    return out


def _log_bloat(learning_id: object, old: str, new: str) -> None:
    """Log-only: a merged statement much longer than the old one (statement bloat, FINDINGS 2.2)."""
    if len(new) > max(2 * len(old), len(old) + 300):
        log.info("merged statement of #%s grew from %d to %d characters", learning_id, len(old),
                 len(new))


def _str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


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
    deprecate_blocked: int = 0  # retirements refused (no verifiable contradiction); stored as new
    # `supersede`s refused by the same check: the target stays live and the candidate goes through
    # the near-duplicate check as a `new` one, so it is also counted in ``created``,
    # ``dedup_dropped`` or ``dedup_merged``.
    supersede_blocked: int = 0
    # Filing rules v2 only. Merges (a decision `update` or a near-duplicate merge) refused by the
    # headline backstop because the new statement no longer begins with the old first sentence. A
    # refused update goes through the near-duplicate check as a `new` candidate (one focused second
    # attempt, not counted in ``update_blocked``); a refused near-duplicate merge is dropped.
    headline_blocked: int = 0
    # Filing rules v2 only. Merges the merge check flagged as adding a different rule than the old
    # lesson's main rule (topic-match merges). By default they are applied and counted in
    # ``offrule_flagged``. With ``judge_offrule_block`` they are refused and counted in
    # ``offrule_blocked``: the candidate is stored as new on both paths, so it is also counted in
    # ``created`` (and, for a decision `update`, in ``update_blocked``).
    offrule_blocked: int = 0
    offrule_flagged: int = 0
    dedup_dropped: int = 0  # `new` candidates found to restate an existing lesson; not stored
    dedup_merged: int = 0  # `new` candidates that added a fact; merged into the existing lesson
    passes: int = 0  # extract passes run (a long session is split into several)
    # Judge replies that were unusable (cut off at the cap, runaway timeout, not JSON). Each failed
    # safe: a decision -> stored as new, a check -> refused, an extract -> pass left for next run.
    unusable_replies: int = 0
    failed_passes: int = 0  # extract passes whose reply was unusable; their turns stay queued
    skipped_small: bool = False  # below the one-off gate; turns left un-consolidated to accumulate
    chunks_consolidated: int = 0
    learning_ids: list[int] = field(default_factory=list)


def extract_learnings(
    chat: ChatModel, turns: list[dict], judge: JudgeConfig | None = None
) -> list[dict]:
    """Run the extract pass over ``turns`` (``[{role, text, line}]``); return candidate lessons."""
    if not turns:
        return []
    j = _resolve(judge)
    convo = "\n\n".join(f"{t.get('role', '?').upper()}: {t.get('text', '')}" for t in turns)
    reply = chat.complete_json(j.prompts.extract_system, convo, schema=j.prompts.extract_schema)
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
    dedup_min_similarity: float = DEDUP_MIN_SIMILARITY,
    judge: JudgeConfig | None = None,
) -> None:
    """Apply one candidate as new / update / supersede / deprecate / drop against existing lessons.

    ``deprecate`` and ``drop`` are the retirement half of the loop: the first retires an existing
    lesson the session proved wrong (storing no replacement), the second discards a candidate that
    merely re-learns an already-retired lesson, so retirement is not silently undone next session.
    Both are counted on the result rather than logged away, since a candidate that vanishes without
    a trace is indistinguishable from one that was never extracted.

    ``supersede`` and ``deprecate`` retire a live lesson only after the judge quotes, verbatim, the
    span of the candidate that contradicts it (:func:`_contradiction_verified`). A refused
    ``deprecate`` stores the candidate as new. A refused ``supersede`` is treated as a ``new``
    candidate *including* the near-duplicate check: the judge linked it to the target and found no
    contradiction, so it most likely restates or extends that lesson and should be dropped or
    merged rather than stored as a blind duplicate.

    Filing rules v2 (``judge``) add two refusals of an ``update``: the headline backstop (the merged
    statement no longer begins with the target's first sentence) sends the candidate through the
    near-duplicate check like a refused supersede; an off-rule merge (it adds another rule) stores
    the candidate as new. Judge-written statements and details are also cleaned
    (:func:`_clean_text`).
    """
    j = _resolve(judge)
    matches, sims = _nearest_learnings(store, embedder, candidate["statement"], scope)
    decision = (_decide(chat, candidate, matches, sims, result, judge=j) if matches
                else {"action": "new"})
    # True once a refused update/deprecate falls through to "store as new" (no near-duplicate check;
    # a refused supersede does not set it, so its candidate still gets that check).
    fallback = False
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
    if j.v2:
        statement = _clean_text(statement) or candidate["statement"]
        detail = _clean_text(detail) or candidate.get("detail")
    # Rescale here, at the model boundary: the judge is asked for 0..1 but sometimes answers on a
    # 1–5/1–10 scale, and importance is a ranking weight — an unscaled 9.0 buries every relevant
    # lesson. `add_learning` clamped, but the `update` path wrote the raw number straight through.
    importance = normalize_importance(decision.get("importance", candidate.get("importance", 0.5)))
    target_id = decision.get("target_id")

    if action == "drop":
        retired = [m.learning_id for m in matches if not m.is_live]
        if retired:
            log.info("dropped candidate %r: re-learns retired lesson(s) %s",
                     candidate["statement"][:80], retired)
        else:
            # `drop` is defined for retired lessons, but the judge also uses it for a candidate
            # that restates a live lesson; say so instead of "re-learns retired (none matched)".
            live = [m.learning_id for m in matches if m.is_live][:3]
            log.info("dropped candidate %r: judge found it already covered (no retired match; "
                     "nearest live lesson(s) %s)", candidate["statement"][:80], live)
        result.dropped += 1
        return

    if action == "deprecate" and _valid_target(target_id, matches):
        target = next(m for m in matches if m.learning_id == target_id)
        if _contradiction_verified(chat, target, candidate, result, judge=j):
            reason = decision.get("reason") or f"contradicted while learning: {statement}"
            deprecate_learning(store, target_id, reason=reason)
            log.info("deprecated learning #%s: %s", target_id, reason)
            result.deprecated += 1
            return
        # Refused: keep the old lesson live and store the candidate in its own words.
        log.info("blocked retirement of learning #%s: no verifiable contradiction; storing "
                 "candidate %r as new", target_id, candidate["statement"][:120])
        result.deprecate_blocked += 1
        fallback = True
        statement = candidate["statement"]
        detail = candidate.get("detail")
        importance = normalize_importance(candidate.get("importance", 0.5))
        action = "new"

    if action == "update" and _valid_target(target_id, matches):
        target = next(m for m in matches if m.learning_id == target_id)
        keeps, kind, lost = _merge_check(chat, target, statement, detail, candidate, result,
                                         judge=j)
        if keeps:
            if j.v2:
                _log_bloat(target_id, target.statement, statement)
            # Lesson template: extra sentences of the merged statement lead the detail.
            base = detail if detail is not None else target.detail
            statement, templated = one_sentence(statement, base)
            # Only pass what the merge actually produced: in `update_learning` a ``None`` *clears*
            # the column, so an absent detail/anchors must be omitted rather than sent as None.
            patch: dict = {"statement": statement, "importance": importance}
            if detail is not None or templated != base:
                patch["detail"] = templated
            if source_anchors:
                patch["source_anchors"] = json.dumps(source_anchors)
            store.update_learning(target_id, **patch, revision_reason="consolidate:update")
            reembed_learning(store, embedder, target_id)
            result.updated += 1
            result.learning_ids.append(target_id)
            return
        if kind == "headline":
            # v2: the judge rewrote the target's first sentence. Not a fallback: the candidate
            # gets one focused second attempt through the near-duplicate check below, whose merge
            # must pass the same backstop; otherwise it is stored in its own words.
            log.info("blocked update of learning #%s: the merged statement no longer begins with "
                     "its first sentence; checking candidate %r as new", target_id,
                     candidate["statement"][:120])
            result.headline_blocked += 1
        else:
            # Refused: keep the target intact and store the candidate in its *own* words (the
            # judge's merged text is a blend of target + candidate and would duplicate the target).
            if kind == "off_rule":
                log.info("blocked update of learning #%s (%s); storing candidate %r as new",
                         target_id, lost, candidate["statement"][:120])
                result.offrule_blocked += 1
            else:
                log.info(
                    "blocked update of learning #%s (merge would lose: %s); storing candidate %r "
                    "as new",
                    target_id,
                    lost or "its core claim",
                    candidate["statement"][:120],
                )
            result.update_blocked += 1
            fallback = True
        statement = candidate["statement"]
        detail = candidate.get("detail")
        importance = normalize_importance(candidate.get("importance", 0.5))
        action = "new"

    if action == "supersede" and _valid_target(target_id, matches):
        target = next(m for m in matches if m.learning_id == target_id)
        if not _contradiction_verified(chat, target, candidate, result, judge=j):
            # Refused: keep the old lesson live. No `fallback` on purpose — the candidate goes
            # through the near-duplicate check below, which drops or merges it if it restates a
            # live lesson (normally the target), and otherwise stores it in its own words (not the
            # judge's replacement text, which was written to stand in for the target).
            log.info("blocked supersede of learning #%s: no verifiable contradiction; checking "
                     "candidate %r as new", target_id, candidate["statement"][:120])
            result.supersede_blocked += 1
            statement = candidate["statement"]
            detail = candidate.get("detail")
            importance = normalize_importance(candidate.get("importance", 0.5))
            action = "new"

    if action != "supersede" and not fallback and dedup_min_similarity > 0:
        near = [m for m in matches
                if m.is_live and sims.get(m.learning_id, 0.0) >= dedup_min_similarity]
        if near:
            best = max(near, key=lambda m: sims[m.learning_id])
            if _dedup_into(store, embedder, chat, best, candidate, source_anchors, result,
                           judge=j):
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
    min_turns: int = 0,
    min_chars: int = 0,
    dedup_min_similarity: float = DEDUP_MIN_SIMILARITY,
    judge: JudgeConfig | None = None,
) -> ConsolidateResult:
    """Distil one chat document's un-consolidated turns into learnings; idempotent on re-run.

    Long sessions are read in several passes (``_MAX_TURNS`` turns / ``_MAX_PASS_CHARS``
    characters each), and each pass's turns are marked consolidated only once that pass is done —
    previously only the first 200 turns were read while *all* were marked, so the rest of a long
    session was silently never distilled. ``min_turns``/``min_chars`` (the CLI passes the
    ``consolidate_min_*`` settings) skip sessions too small to hold a durable lesson, without
    marking them, so a resumed session's turns accumulate until they clear the gate.

    ``judge`` selects prompts, short-call cap and filing rules (the CLI passes
    ``judge_config(settings)``). ``None`` = today's module constants, read when the session starts.
    """
    result = ConsolidateResult()
    chunks = store.unconsolidated_chat_chunks(doc_id)
    if not chunks:
        return result
    if len(chunks) < min_turns or sum(len(h.text) for h in chunks) < min_chars:
        log.info("skipping doc %s: %d turns below the one-off gate", doc_id, len(chunks))
        result.skipped_small = True
        return result
    j = _resolve(judge)
    log.debug("judge: filing rules %s, reasoning field %s, short cap %s", j.filing_rules,
              j.reasoning_field, j.short_max_tokens)
    path = chunks[0].path
    windows = _windows(chunks)
    log.info("consolidating doc %s (%s): %d turns in %d pass(es)", doc_id, path, len(chunks),
             len(windows))
    for p, window in enumerate(windows, 1):
        turns = [{"role": h.symbol or "?", "text": h.text, "line": h.start_line} for h in window]
        result.turns_read += len(turns)
        result.passes += 1
        try:
            candidates = extract_learnings(chat, turns, judge=j)
        except ChatReplyError as exc:
            # Not marked consolidated: an unusable extract must not look like "no lessons here".
            # The turns stay queued for the next run (the judge is not deterministic at temp 0).
            log.warning("doc %s pass %d/%d: extract reply unusable (%s); turns left for the next "
                        "run", doc_id, p, len(windows), exc)
            result.unusable_replies += 1
            result.failed_passes += 1
            continue
        result.candidates += len(candidates)
        log.info("doc %s pass %d/%d: %d candidate(s)", doc_id, p, len(windows), len(candidates))
        for i, cand in enumerate(candidates, 1):
            log.info("doc %s pass %d candidate %d/%d: %r", doc_id, p, i, len(candidates),
                     cand["statement"][:80])
            anchors = [{"transcript_path": path, "line": t["line"]} for t in turns[:3]]
            consolidate_candidate(
                store, embedder, chat, cand, scope=scope, source_anchors=anchors, result=result,
                dedup_min_similarity=dedup_min_similarity, judge=j,
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
) -> tuple[list[Learning], dict[int, float]]:
    """Nearest ``kind='learning'`` matches: up to ``_MATCH_POOL`` live + ``_MATCH_RETIRED`` retired.

    Retired lessons are fetched with ``include_retired=True`` *and* budgeted separately, so showing
    them to the judge can never displace a live match it might have merged into.
    """
    [vec] = embedder.embed([statement])
    over_fetch = (_MATCH_POOL + _MATCH_RETIRED) * 2  # kind/scope filtering trims the pool
    hits = store.search_vec(vec, k=over_fetch, kind="learning", include_retired=True)
    live: list[Learning] = []
    retired: list[Learning] = []
    sims: dict[int, float] = {}
    for h in hits:
        learning = store.learning_by_doc_id(h.doc_id)
        if learning is None:
            continue
        sims.setdefault(learning.learning_id, round(1.0 - float(h.distance), 3))
        if scope is not None and learning.scope not in (scope, None):
            continue
        if learning.is_live:
            if len(live) < _MATCH_POOL:
                live.append(learning)
        elif len(retired) < _MATCH_RETIRED:
            retired.append(learning)
    return live + retired, sims


def _ask(
    chat: ChatModel, system: str, user: str, schema: dict, *, what: str,
    result: ConsolidateResult | None, judge: JudgeConfig | None = None,
) -> dict | None:
    """One short judge call, capped at ``judge.short_max_tokens`` (``chat_short_max_tokens``;
    without a ``judge``, ``SHORT_REPLY_MAX_TOKENS``). ``None`` if the reply was unusable, so the
    caller can fail safe. A judge outage (:class:`ChatBackendError`) propagates."""
    cap = judge.short_max_tokens if judge is not None else SHORT_REPLY_MAX_TOKENS
    try:
        return chat.complete_json(system, user, schema=schema, max_tokens=cap)
    except ChatReplyError as exc:
        log.warning("%s: judge reply unusable (%s); failing safe", what, exc)
        if result is not None:
            result.unusable_replies += 1
        return None


def _decide(
    chat: ChatModel,
    candidate: dict,
    matches: list[Learning],
    sims: dict[int, float] | None = None,
    result: ConsolidateResult | None = None,
    *,
    judge: JudgeConfig | None = None,
) -> dict:
    j = _resolve(judge)
    describe = _describe_match_labelled if j.listing == "labelled" else _describe_match
    listing = "\n".join(describe(m, (sims or {}).get(m.learning_id)) for m in matches)
    user = (
        f"NEW candidate lesson:\n[{candidate['type']}] {candidate['statement']}"
        + (f"\ndetail: {candidate['detail']}" if candidate.get("detail") else "")
        + f"\n\nEXISTING similar lessons:\n{listing}"
    )
    decision = _ask(chat, j.prompts.consolidate_system, user, j.prompts.consolidate_schema,
                    what=f"decision for {candidate['statement'][:60]!r}", result=result, judge=j)
    return decision if decision is not None else {"action": "new"}  # unusable -> store as new


def _lesson_text(statement: str, detail: str | None) -> str:
    return statement + (f"\ndetail: {detail}" if detail else "")


def _merge_check(
    chat: ChatModel,
    target: Learning,
    statement: str,
    detail: str | None,
    candidate: dict,
    result: ConsolidateResult | None = None,
    *,
    judge: JudgeConfig | None = None,
) -> tuple[bool, str, str]:
    """Does the merged text still state ``target``'s core claim? Returns ``(ok, kind, why)``.

    Strict: the judge must quote where the old claim survives, and the quote is checked against
    the merged text in code (a bare "preserves: true" is not enough — a lenient judge approved
    every merge it was shown). Facts in the merge that come from neither the old lesson nor the
    candidate are logged as a warning (not a block). Fails safe: an unusable reply counts as *not*
    preserved.

    ``kind`` says why a merge was refused: "unusable" (the reply), "claim" (not preserved / no
    verbatim evidence). Filing rules v2 refuse, in this order: "headline" — :func:`headline_kept`
    fails, decided in code *before* the judge is asked; "unusable"; "claim"; "off_rule" — the judge
    lists a sentence or clause of NEW that states a different rule than OLD MAIN RULE, refused only
    with ``judge_offrule_block`` (default: applied, logged and counted in ``offrule_flagged``; in
    validation the block refused merges that kept the headline and should have gone through). Under
    v2 the judge is told OLD MAIN RULE (:func:`main_rule`) instead of choosing OLD's core claim
    itself, and lost facts are logged, not blocked.
    """
    j = _resolve(judge)
    lid = getattr(target, "learning_id", "?")
    if j.v2 and not headline_kept(target.statement, statement):
        return False, "headline", "the merged statement no longer begins with its first sentence"
    new_text = _lesson_text(statement, detail)
    old_text = _lesson_text(target.statement, target.detail)
    cand_text = _lesson_text(candidate["statement"], candidate.get("detail"))
    if j.v2:
        user = (f"OLD:\n{old_text}\n\nOLD MAIN RULE:\n{main_rule(target.statement)}"
                f"\n\nNEW:\n{new_text}\n\nCANDIDATE:\n{cand_text}")
    else:
        user = f"OLD:\n{old_text}\n\nNEW:\n{new_text}\n\nCANDIDATE:\n{cand_text}"
    reply = _ask(chat, j.prompts.preserve_system, user, j.prompts.preserve_schema,
                 what=f"merge check into #{lid}", result=result, judge=j)
    if reply is None:
        return False, "unusable", "the merge check reply was unusable"
    norm = _norm_v2 if j.v2 else _norm
    evidence = str(reply.get("evidence_in_new") or "").strip()
    quoted = bool(evidence) and norm(evidence) in norm(new_text)
    additions = [str(a) for a in reply.get("unsupported_additions") or [] if str(a).strip()]
    if reply.get("preserves") is not True:
        return False, "claim", str(reply.get("lost") or "its core claim")
    if not quoted:
        return (False, "claim",
                "no verbatim evidence that the merged text still states the old claim")
    if j.v2:
        off_rule = _str_list(reply.get("off_rule_additions"))
        if off_rule:
            # Topic-match merges (R2 #1761, P-S3 #538/#1365/#1035): the old headline survives, so
            # the quote check passes, but the merge appends a different rule. Blocking is opt-in:
            # the PR #39 validation showed it refusing good merges (p05 -> #1911, cross-account
            # wash -> #1933), which then became near-duplicate lessons.
            why = "the merge adds another rule: " + "; ".join(off_rule[:3])
            if j.offrule_block:
                return False, "off_rule", why
            log.warning("merge into #%s: %s (applied; judge_offrule_block refuses these)", lid, why)
            if result is not None:
                result.offrule_flagged += 1
        lost_facts = _str_list(reply.get("lost_facts"))
        if lost_facts:
            # Logged, not blocked: blocking on lost facts is unmeasured (see below for additions).
            log.warning("merge into #%s drops facts of the old lesson: %s", lid,
                        "; ".join(lost_facts[:3]))
    if additions:
        # Logged, not blocked: in a blind review, blocking on additions turned most merges into
        # near-duplicate NEW lessons (dedup fell 3.67 -> 2.83). The verbatim quote above is what
        # actually catches a merge that overwrote the old lesson.
        log.warning("merge into #%s adds facts found in neither lesson: %s",
                    target.learning_id if hasattr(target, "learning_id") else "?",
                    "; ".join(additions[:3]))
    return True, "", ""


def _merge_preserves(
    chat: ChatModel,
    target: Learning,
    statement: str,
    detail: str | None,
    candidate: dict,
    result: ConsolidateResult | None = None,
    *,
    judge: JudgeConfig | None = None,
) -> tuple[bool, str]:
    """:func:`_merge_check` without the refusal kind: ``(preserves, why_not)``."""
    ok, _kind, why = _merge_check(chat, target, statement, detail, candidate, result, judge=judge)
    return ok, why


def _dedup_into(
    store: Store,
    embedder: Embedder,
    chat: ChatModel,
    existing: Learning,
    candidate: dict,
    source_anchors: list[dict] | None,
    result: ConsolidateResult,
    *,
    judge: JudgeConfig | None = None,
) -> bool:
    """Ask whether ``candidate`` restates ``existing``; drop or merge it. True = handled (not new).

    Not the same lesson (or an unusable reply) -> False, and the candidate is stored as new, so a
    broken check never loses a lesson. Same lesson adding nothing -> dropped. Same lesson adding a
    fact -> merged into ``existing`` through the verbatim-quote merge check; if that merge would
    lose the existing claim, the candidate is dropped and ``existing`` stays intact. An unusable
    merge-check reply (cut off at the cap) -> False as well: stored as new, never dropped.

    Filing rules v2: the judge is shown EXISTING MAIN RULE; a merge refused by the headline backstop
    is dropped like any lost claim, but one refused as off-rule (it appends a different rule) means
    the candidate was not the same lesson after all -> False, stored as new, never dropped.
    """
    j = _resolve(judge)
    existing_text = _lesson_text(existing.statement, existing.detail)
    cand_text = _lesson_text(candidate["statement"], candidate.get("detail"))
    if j.v2:
        user = (f"EXISTING:\n{existing_text}\n\nEXISTING MAIN RULE:\n"
                f"{main_rule(existing.statement)}\n\nCANDIDATE:\n{cand_text}")
    else:
        user = f"EXISTING:\n{existing_text}\n\nCANDIDATE:\n{cand_text}"
    reply = _ask(chat, j.prompts.dedup_system, user, j.prompts.dedup_schema,
                 what=f"dedup check against #{existing.learning_id}", result=result, judge=j)
    if reply is None or reply.get("same_lesson") is not True:
        return False
    stmt = str(reply.get("merged_statement") or "").strip()
    detail = str(reply.get("merged_detail") or "").strip() or None
    if j.v2:
        stmt = _clean_text(stmt)
        detail = _clean_text(detail) or None
    if reply.get("adds_new_fact") is not True or not stmt:
        log.info("dropped candidate %r: restates learning #%s",
                 candidate["statement"][:80], existing.learning_id)
        result.dedup_dropped += 1
        return True
    keeps, kind, why = _merge_check(chat, existing, stmt, detail, candidate, result, judge=j)
    if not keeps:
        if kind == "unusable":
            # A broken check never loses a lesson: the judge said the candidate adds a fact, and
            # a cut-off/garbled merge check says nothing about the merge. Before 2026-11-05 this
            # fell through to "dropped" (the eval lost 3 candidates per judge on long lessons).
            log.info("kept candidate %r as new: the merge check into #%s was unusable",
                     candidate["statement"][:80], existing.learning_id)
            return False
        if kind == "off_rule":
            log.info("kept candidate %r as new: its merge into #%s was refused (%s)",
                     candidate["statement"][:80], existing.learning_id, why)
            result.offrule_blocked += 1
            return False
        if kind == "headline":
            result.headline_blocked += 1
        log.info("dropped candidate %r: same as #%s but the merge would lose its claim (%s)",
                 candidate["statement"][:80], existing.learning_id, why)
        result.dedup_dropped += 1
        return True
    if j.v2:
        _log_bloat(existing.learning_id, existing.statement, stmt)
    # Lesson template: extra sentences of the merged statement lead the detail.
    base = detail if detail else existing.detail
    stmt, templated = one_sentence(stmt, base)
    patch: dict = {"statement": stmt}
    if detail or templated != base:
        patch["detail"] = templated
    if source_anchors:
        patch["source_anchors"] = json.dumps(source_anchors)
    store.update_learning(existing.learning_id, **patch, revision_reason="consolidate:dedup-merge")
    reembed_learning(store, embedder, existing.learning_id)
    result.updated += 1
    result.dedup_merged += 1
    result.learning_ids.append(existing.learning_id)
    return True


# Filing rules v2: the shortest ``false_in_old`` quote the retirement check accepts, in words.
MIN_FALSE_IN_OLD_WORDS = 3


def _contradiction_verified(
    chat: ChatModel,
    target: Learning,
    candidate: dict,
    result: ConsolidateResult | None = None,
    *,
    judge: JudgeConfig | None = None,
) -> bool:
    """True only if the judge quotes a span of the candidate that contradicts ``target``.

    The gate for both retiring actions, ``deprecate`` and ``supersede``. The quote is checked
    against the candidate's text in code; an unusable reply fails safe (the lesson stays live).

    Filing rules v2 need a second quote, anchored in the old lesson: ``false_in_old``, the words of
    OLD MAIN RULE (:func:`main_rule`, shown to the judge) that the candidate makes false. Both
    quotes must be verbatim (after :func:`_norm_v2`) — the first in the candidate, the second in
    the old main rule — so a real but non-contradicting candidate quote (#1852, #1142) no longer
    retires a lesson whose main rule still holds. ``false_in_old`` must also be at least
    :data:`MIN_FALSE_IN_OLD_WORDS` words (or the whole main rule, if shorter): a one-word quote
    ("vLLM-only") is found in almost any rule and let a wrong retirement through (#2030 -> #2034).
    """
    j = _resolve(judge)
    cand_text = _lesson_text(candidate["statement"], candidate.get("detail"))
    old_text = _lesson_text(target.statement, target.detail)
    rule = main_rule(target.statement)
    if j.v2:
        user = f"OLD:\n{old_text}\n\nOLD MAIN RULE:\n{rule}\n\nCANDIDATE:\n{cand_text}"
    else:
        user = f"OLD:\n{old_text}\n\nCANDIDATE:\n{cand_text}"
    reply = _ask(chat, j.prompts.deprecate_system, user, j.prompts.deprecate_schema,
                 what=f"retirement check of #{target.learning_id}", result=result, judge=j)
    if reply is None:
        return False  # unusable -> the lesson stays live
    quote = str(reply.get("contradiction_quote") or "").strip()
    if not j.v2:
        return (reply.get("contradicts") is True and bool(quote)
                and _norm(quote) in _norm(cand_text))
    false_in_old = str(reply.get("false_in_old") or "").strip()
    min_words = min(MIN_FALSE_IN_OLD_WORDS, len(_norm_v2(rule).split()))
    long_enough = len(_norm_v2(false_in_old).split()) >= max(min_words, 1)
    ok = (reply.get("contradicts") is True
          and bool(quote) and _norm_v2(quote) in _norm_v2(cand_text)
          and bool(false_in_old) and _norm_v2(false_in_old) in _norm_v2(rule)
          and long_enough)
    if not ok and reply.get("contradicts") is True:
        log.info("retirement check of #%s: quotes not verifiable or too short (candidate %r, old "
                 "main rule %r)", target.learning_id, quote[:80], false_in_old[:80])
    return ok


def _norm(s: str) -> str:
    """Case/whitespace/quote-insensitive form for the verbatim-evidence check."""
    return " ".join(s.lower().replace("`", "").replace('"', "").replace("'", "").split())


def _describe_match(m: Learning, sim: float | None = None) -> str:
    """One prompt line for a candidate's neighbour, flagged ``[RETIRED: why]`` when not live.

    Similarity and provenance help the judge tell a restatement from a related-but-different lesson
    (eval: without them it re-filed existing lessons as `new`).
    """
    line = f"- id={m.learning_id} [{m.type}] ({'; '.join(_match_meta(m, sim))}) {m.statement}"
    if m.detail:
        line += f" ({m.detail})"
    if not m.is_live:
        line += f"  [RETIRED: {_retired_why(m)}]"
    return line


def _match_meta(m: Learning, sim: float | None) -> list[str]:
    meta = [f"similarity {sim:.2f}"] if sim is not None else []
    meta.append(f"created {str(m.created_at)[:10]}")
    if str(m.updated_at)[:10] != str(m.created_at)[:10]:
        meta.append(f"updated {str(m.updated_at)[:10]}")
    if m.reuse_count:
        meta.append(f"recalled {m.reuse_count}x")
    return meta


def _retired_why(m: Learning) -> str:
    return m.deprecated_reason or (
        f"superseded by #{m.superseded_by}" if m.superseded_by else "retired"
    )


def _describe_match_labelled(m: Learning, sim: float | None = None) -> str:
    """``judge_listing = "labelled"``: one labelled field per line instead of one long line.

    ``main rule:`` is computed in code (:func:`main_rule`), as the other v2 calls already get it
    (OLD MAIN RULE / EXISTING MAIN RULE); the judge no longer has to find the first sentence inside
    an unlabelled line where the metadata and the detail are both in parentheses (eval 2026-11-06:
    Nemotron 3.5 Lightning copied the detail, a later sentence or a tag-led title as the headline).
    ``statement:`` is shown only when it says more than the main rule, so a one-sentence lesson is
    not repeated; the detail and a retirement get their own lines.
    """
    rule = main_rule(m.statement)
    lines = [f"- id={m.learning_id} [{m.type}] ({'; '.join(_match_meta(m, sim))})",
             f"  main rule: {rule}"]
    statement = strip_type_tags(m.statement)
    if _norm_v2(statement) != _norm_v2(rule):
        lines.append(f"  statement: {statement}")
    if m.detail:
        lines.append(f"  detail: {m.detail}")
    if not m.is_live:
        lines.append(f"  [RETIRED: {_retired_why(m)}]")
    return "\n".join(lines)


def _valid_target(target_id: object, matches: list[Learning]) -> bool:
    """A target is actionable only if the judge was actually shown it **and** it is still live.

    Retired lessons appear in the prompt for context; editing or re-retiring one would undo a
    deliberate retirement, so they are never valid targets. An invalid target falls through to the
    insert path — a contradictory decision costs a possible duplicate, never a lost candidate.
    """
    return isinstance(target_id, int) and any(
        m.learning_id == target_id and m.is_live for m in matches
    )
