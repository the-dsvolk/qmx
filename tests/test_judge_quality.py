"""Judge quality fixes: multi-pass long sessions, the one-off gate, the strict merge check."""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import _MAX_PASS_CHARS, consolidate_session
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


def _doc(store, embedder, tmp_path, turns: list[str]):
    lines = [json.dumps({"type": "user" if i % 2 == 0 else "assistant",
                         "message": {"role": "user" if i % 2 == 0 else "assistant", "content": t}})
             for i, t in enumerate(turns)]  # fmt: skip
    path = tmp_path / "sess.jsonl"
    path.write_text("\n".join(lines))
    index_transcript(path, store, embedder)
    return store.document_id("chat", str(path.resolve()))


def test_long_session_is_read_in_several_passes_and_fully_marked(store, tmp_path):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path, [f"turn {i}: distinct content {i * 7}" for i in range(450)])
    chat = FakeChat(extractions=[[{"type": "howto", "statement": f"lesson from pass {p}"}]
                                 for p in range(3)])  # fmt: skip
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    assert res.passes == 3  # 200 + 200 + 50 turns
    assert res.turns_read == res.chunks_consolidated == 450
    assert res.created == 3  # every pass was actually read
    assert s.unconsolidated_chat_chunks(doc) == []


def test_character_budget_also_splits_a_pass(store, tmp_path):
    s, embedder = store
    turn = "x" * 1400
    n = _MAX_PASS_CHARS // 1400 + 10  # well under 200 turns, over the char budget
    doc = _doc(s, embedder, tmp_path, [f"{i} {turn}" for i in range(n)])
    res = consolidate_session(s, embedder, FakeChat(), doc, scope=SCOPE)
    assert res.passes == 2


def test_one_off_gate_skips_without_marking(store, tmp_path):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path, ["which chat made this doc?", "it was session X"])
    chat = FakeChat(extractions=[[{"type": "howto", "statement": "trivia"}]])
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE, min_turns=8, min_chars=4000)
    assert res.skipped_small and res.passes == 0 and res.created == 0
    assert len(s.unconsolidated_chat_chunks(doc)) == 2  # left to accumulate
    # Without the gate (library default) the same session is processed.
    res2 = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    assert res2.passes == 1 and res2.created == 1


def _update(store, tmp_path, check: dict):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path, ["check wash sales everywhere", "store lots per account"])
    target = add_learning(s, embedder, type="howto", scope=SCOPE,
                          statement="Track wash-sale freezes across all household accounts")
    chat = FakeChat(
        extractions=[[{"type": "howto", "statement": "Wash sales span IRAs too",
                       "detail": "an IRA buy can trigger it"}]],
        decisions=[{"action": "update", "target_id": target,
                    "statement": "Track wash-sale freezes across all household accounts, "
                                 "including IRAs", "detail": "an IRA buy can trigger it"}],
        checks=[check],
    )  # fmt: skip
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    return s, target, res


def test_strict_check_accepts_a_real_quote(store, tmp_path):
    _, _, res = _update(store, tmp_path, {
        "preserves": True, "unsupported_additions": [], "lost": "",
        "evidence_in_new": "track WASH-SALE freezes across all household accounts"})
    assert (res.updated, res.update_blocked) == (1, 0)


def test_strict_check_rejects_preserves_true_without_a_real_quote(store, tmp_path):
    s, target, res = _update(store, tmp_path, {
        "preserves": True, "unsupported_additions": [], "lost": "",
        "evidence_in_new": "household rule kept"})  # not actually in NEW
    assert (res.updated, res.update_blocked, res.created) == (0, 1, 1)
    kept = s.get_learning(target).statement
    assert kept == "Track wash-sale freezes across all household accounts"


def test_unsupported_additions_warn_but_do_not_block(store, tmp_path, caplog):
    _, _, res = _update(store, tmp_path, {
        "preserves": True, "lost": "",
        "evidence_in_new": "Track wash-sale freezes across all household accounts",
        "unsupported_additions": ["E*TRADE never reports wash sales"]})
    assert (res.updated, res.update_blocked) == (1, 0)
    assert "adds facts found in neither lesson" in caplog.text


def _deprecate(store, tmp_path, retire_check: dict | None):
    s, embedder = store
    doc = _doc(s, embedder, tmp_path, ["cursor sdk test", "inline OAuth servers lose saved logins"])
    target = add_learning(s, embedder, type="howto", scope=SCOPE,
                          statement="Load OAuth MCP servers via setting_sources, not inline")
    chat = FakeChat(
        extractions=[[{"type": "howto", "statement": "Store the Cursor API key in the Keychain"}]],
        decisions=[{"action": "deprecate", "target_id": target, "reason": "obsolete"}],
        retire_checks=[retire_check] if retire_check else None,
    )
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    return s, target, res


def test_retirement_needs_a_real_contradiction(store, tmp_path):
    s, target, res = _deprecate(store, tmp_path, {"contradicts": False, "contradiction_quote": ""})
    assert (res.deprecated, res.deprecate_blocked, res.created) == (0, 1, 1)
    assert s.get_learning(target).is_live  # the correct lesson stays live


def test_retirement_quote_must_be_in_the_candidate(store, tmp_path):
    _, _, res = _deprecate(store, tmp_path, {
        "contradicts": True, "contradiction_quote": "setting_sources no longer works"})
    assert (res.deprecated, res.deprecate_blocked) == (0, 1)


def test_verified_contradiction_retires(store, tmp_path):
    s, target, res = _deprecate(store, tmp_path, None)  # default fake quotes the candidate
    assert (res.deprecated, res.deprecate_blocked, res.created) == (1, 0, 0)
    assert not s.get_learning(target).is_live
