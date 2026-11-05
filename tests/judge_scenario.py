"""A scripted consolidation run that exercises all five judge calls, recorded request by request.

Used to pin that the default judge settings send exactly what qmx sent before filing rules v2
(main @ 98bfd69): the same system prompts, user messages, schemas and caps, in the same order, and
the same resulting store. ``GOLDEN_SHA256`` was computed on that commit with ``judge`` unset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from qmx.consolidate import consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

GOLDEN_SHA256 = "3d01ca183119a795c9ccc271c642a35b8b8d676743692e715f9ef6e5f6136067"

SCOPE = "the-dsvolk/qmx"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

T1 = ("Raise IAM pull requests at project level, never bucket level",
      "bucket-level IAM bindings failed review twice")
T2 = ("Store API secrets in a restricted local directory such as the secret project folder",
      "chmod 700 keeps other users out")
T3 = ("Pin the embedding model tag in config so reindexing stays reproducible",
      "a floating tag silently changed vectors")
T4 = ("Ollama keeps the judge model resident for five minutes after each call",
      "keep_alive defaults to 5m")
T5 = ("Track wash-sale freezes across all household brokerage accounts",
      "spouse, IRAs and RSUs count as one entity")

CANDIDATES = [
    {"type": "howto", "statement": "Raise IAM pull requests at project level with one owner",
     "detail": "name a single owning team", "importance": 0.7},
    {"type": "howto", "statement": T2[0] + " folder", "detail": "chmod 700 the folder",
     "importance": 0.6},
    {"type": "decision", "statement": "Pin the embedding model digest, not the tag, in config",
     "detail": "tags can be re-pushed", "importance": 0.8},
    {"type": "mistake", "statement": "Ollama judge model resident time is configurable per call",
     "detail": "keep_alive=-1 pins it", "importance": 0.5},
    {"type": "howto", "statement": "Store wash-sale lots and trades in separate tables",
     "detail": "lots are snapshots, trades are events", "importance": 0.6},
]


class Recorder:
    """Wraps a ChatModel and records every request exactly as the judge would receive it."""

    def __init__(self, inner: FakeChat) -> None:
        self.inner = inner
        self.calls: list[list] = []

    def complete_json(self, system, user, schema=None, *, max_tokens=None):
        self.calls.append([system, _DATE.sub("DATE", user), json.dumps(schema), max_tokens])
        return self.inner.complete_json(system, user, schema, max_tokens=max_tokens)


def run_scenario(tmp_path: Path, **consolidate_kwargs) -> tuple[list[list], list[list]]:
    """Run the script; return (recorded requests, final lesson rows)."""
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        ids = [add_learning(s, embedder, type="howto", scope=SCOPE, statement=st, detail=dt)
               for st, dt in (T1, T2, T3, T4, T5)]
        path = tmp_path / "sess.jsonl"
        path.write_text("\n".join(
            json.dumps({"type": r, "message": {"role": r, "content": t}})
            for r, t in (("user", "lessons from the IAM, secrets and wash-sale work"),
                         ("assistant", "noted"))))
        index_transcript(path, s, embedder)
        doc = s.document_id("chat", str(path.resolve()))
        chat = FakeChat(
            extractions=[CANDIDATES],
            decisions=[
                {"reasoning": "same lesson", "action": "update", "target_id": ids[0],
                 "statement": T1[0] + " and one owner", "detail": T1[1] + "; name one owner"},
                {"reasoning": "novel", "action": "new"},
                {"reasoning": "stale", "action": "supersede", "target_id": ids[2],
                 "statement": CANDIDATES[2]["statement"], "detail": "tags can be re-pushed"},
                {"reasoning": "wrong", "action": "deprecate", "target_id": ids[3],
                 "reason": "keep_alive is configurable"},
                {"reasoning": "merge", "action": "update", "target_id": ids[4],
                 "statement": "Separate lots and trades into distinct tables",
                 "detail": "schema for wash-sale logic"},
            ],
            checks=[
                {"preserves": True, "evidence_in_new": T1[0], "unsupported_additions": [],
                 "lost": ""},
                {"preserves": True, "evidence_in_new": T2[0], "unsupported_additions": [],
                 "lost": ""},
                {"preserves": False, "evidence_in_new": "", "unsupported_additions": [],
                 "lost": "household scope"},
            ],
            retire_checks=[
                {"contradicts": True, "contradiction_quote": "not the tag"},
                {"contradicts": False, "contradiction_quote": ""},
            ],
            dup_checks=[
                {"same_lesson": True, "adds_new_fact": True,
                 "merged_statement": T2[0] + ", with chmod 700",
                 "merged_detail": T2[1] + "; chmod 700 the folder"},
            ],
        )
        rec = Recorder(chat)
        res = consolidate_session(s, embedder, rec, doc, scope=SCOPE, **consolidate_kwargs)
        state = sorted([le.learning_id, le.statement, le.detail, le.is_live, le.superseded_by]
                       for le in s.list_learnings(live_only=False))
        counters = [res.created, res.updated, res.superseded, res.deprecated, res.dropped,
                    res.update_blocked, res.deprecate_blocked, res.supersede_blocked,
                    res.dedup_dropped, res.dedup_merged, res.unusable_replies]
        return rec.calls, [*state, counters]


def scenario_digest(tmp_path: Path, **consolidate_kwargs) -> str:
    calls, state = run_scenario(tmp_path, **consolidate_kwargs)
    return hashlib.sha256(json.dumps({"calls": calls, "state": state}).encode()).hexdigest()
