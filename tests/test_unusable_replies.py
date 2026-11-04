"""One unusable judge reply fails safe for that call instead of deferring the run (2026-11-04).

Before: a merge check whose generation looped until the timeout was retried five times and then
deferred all of ``qmx consolidate --all``; a cut-off extract parsed as ``{}`` and its turns were
marked consolidated as if the session had no lessons. Now each call type has a safe default and an
unusable extract leaves its pass queued.
"""

from __future__ import annotations

import json
import logging

import pytest

from qmx import consolidate as C
from qmx.chat import ChatBackendError
from qmx.consolidate import SHORT_REPLY_MAX_TOKENS, consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

SCOPE = "the-dsvolk/qmx"


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _doc(store, embedder, tmp_path, n_turns: int = 2):
    lines = [json.dumps({"type": "user" if i % 2 == 0 else "assistant",
                         "message": {"role": "user" if i % 2 == 0 else "assistant",
                                     "content": f"turn {i}: distinct content {i * 7}"}})
             for i in range(n_turns)]  # fmt: skip
    path = tmp_path / "sess.jsonl"
    path.write_text("\n".join(lines))
    index_transcript(path, store, embedder)
    return store.document_id("chat", str(path.resolve()))


def _existing(s, embedder, statement="wash sales span every household account"):
    return add_learning(s, embedder, type="howto", statement=statement, scope=SCOPE)


def test_unusable_decision_stores_the_candidate_as_new(store, tmp_path):
    s, embedder = store
    _existing(s, embedder)
    cand = {"type": "howto", "statement": "pin uv in CI"}
    chat = FakeChat(extractions=[[cand]], unusable={"decision": 1})
    res = consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert (res.created, res.unusable_replies) == (1, 1)
    assert s.unconsolidated_chat_chunks(_doc(s, embedder, tmp_path)) == []


def test_unusable_merge_check_blocks_the_update(store, tmp_path):
    s, embedder = store
    lid = _existing(s, embedder)
    cand = {"type": "howto", "statement": "RSU vests count as wash-sale buys"}
    chat = FakeChat(extractions=[[cand]], unusable={"merge": 1},
                    decisions=[{"action": "update", "target_id": lid, "statement": "merged"}])
    res = consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert s.get_learning(lid).statement == "wash sales span every household account"  # intact
    assert (res.updated, res.update_blocked, res.created, res.unusable_replies) == (0, 1, 1, 1)


def test_unusable_dedup_check_keeps_the_candidate(store, tmp_path):
    s, embedder = store
    _existing(s, embedder, "same words exactly")
    chat = FakeChat(extractions=[[{"type": "howto", "statement": "same words exactly"}]],
                    unusable={"dedup": 1})
    res = consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert ("dedup", SHORT_REPLY_MAX_TOKENS) in chat.calls  # the check was asked
    assert (res.created, res.dedup_dropped, res.unusable_replies) == (1, 0, 1)


def test_unusable_retirement_check_keeps_the_lesson_live(store, tmp_path):
    s, embedder = store
    lid = _existing(s, embedder)
    chat = FakeChat(extractions=[[{"type": "mistake", "statement": "that rule was wrong"}]],
                    decisions=[{"action": "deprecate", "target_id": lid, "reason": "wrong"}],
                    unusable={"retire": 1})
    res = consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert s.get_learning(lid).is_live
    assert (res.deprecated, res.deprecate_blocked, res.unusable_replies) == (0, 1, 1)


def test_unusable_extract_leaves_its_pass_queued_and_the_rest_runs(store, tmp_path):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path, n_turns=450)  # 3 passes: 200 + 200 + 50 turns
    chat = FakeChat(extractions=[[{"type": "howto", "statement": f"lesson {p}"}] for p in (2, 3)],
                    unusable={"extract": 1})
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    assert (res.passes, res.failed_passes, res.created) == (3, 1, 2)
    assert len(s.unconsolidated_chat_chunks(doc)) == 200  # pass 1 not marked as "no lessons"
    again = consolidate_session(s, embedder, FakeChat(), doc, scope=SCOPE)
    assert again.passes == 1 and s.unconsolidated_chat_chunks(doc) == []


def test_judge_outage_still_defers_without_marking(store, tmp_path):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path)

    class Down(FakeChat):
        def complete_json(self, *a, **kw):
            raise ChatBackendError("unreachable")

    with pytest.raises(ChatBackendError):
        consolidate_session(s, embedder, Down(), doc, scope=SCOPE)
    assert len(s.unconsolidated_chat_chunks(doc)) == 2


def test_short_calls_are_capped_and_extract_uses_the_configured_cap(store, tmp_path):
    s, embedder = store
    lid = _existing(s, embedder)
    chat = FakeChat(extractions=[[{"type": "howto", "statement": "RSU vests are buys"}]],
                    decisions=[{"action": "update", "target_id": lid, "statement": "m"}])
    consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert chat.calls[0] == ("extract", None)
    assert {k for k, _ in chat.calls[1:]} == {"decision", "merge"}
    assert all(cap == SHORT_REPLY_MAX_TOKENS for _, cap in chat.calls[1:])


def test_every_judge_prompt_asks_for_single_quotes_inside_values():
    for prompt in (C.EXTRACT_SYSTEM, C.CONSOLIDATE_SYSTEM, C.PRESERVE_SYSTEM, C.DEPRECATE_SYSTEM,
                   C.DEDUP_SYSTEM):
        assert prompt.endswith(C._QUOTE_RULE)  # noqa: SLF001
    assert "single quotes" in C._QUOTE_RULE  # noqa: SLF001


def test_drop_without_a_retired_match_is_not_reported_as_retired(store, tmp_path, caplog):
    s, embedder = store
    lid = _existing(s, embedder)
    chat = FakeChat(extractions=[[{"type": "howto", "statement": "household wash sales again"}]],
                    decisions=[{"action": "drop"}])
    with caplog.at_level(logging.INFO, logger="qmx.consolidate"):
        res = consolidate_session(s, embedder, chat, _doc(s, embedder, tmp_path), scope=SCOPE)
    assert res.dropped == 1
    msg = next(r.getMessage() for r in caplog.records if "dropped candidate" in r.getMessage())
    assert "already covered" in msg and str(lid) in msg and "retired lesson(s)" not in msg


def test_verbose_log_names_the_session_pass_and_candidate(store, tmp_path, caplog):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path)
    chat = FakeChat(extractions=[[{"type": "howto", "statement": "pin uv in CI"}]])
    with caplog.at_level(logging.INFO, logger="qmx.consolidate"):
        consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"consolidating doc {doc} (" in text
    assert f"doc {doc} pass 1/1: 1 candidate(s)" in text
    assert f"doc {doc} pass 1 candidate 1/1: 'pin uv in CI'" in text
