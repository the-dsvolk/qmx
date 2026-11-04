"""Retirement guardrail on `supersede` (2026-11-04).

`deprecate` already had to quote, verbatim, the part of the candidate that contradicts the old
lesson; `supersede` retires a lesson too but ran unchecked. In a blind end-to-end eval a judge
superseded three lessons the sessions actually supported (#1852, #380, #1142). Now a supersede must
pass the same check; a refused one keeps the target live and the candidate goes through the
near-duplicate check instead of being stored as a blind duplicate.
"""

from __future__ import annotations

import contextlib
import json
import logging

import pytest

from qmx import cli
from qmx.config import Settings
from qmx.consolidate import SHORT_REPLY_MAX_TOKENS, consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

SCOPE = "the-dsvolk/qmx"
TRANSCRIPT = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "user", "message": {"role": "user", "content": "load glean via setting_sources"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "saved login reused"}},
    ]
)
OLD = "Load Cursor OAuth MCP servers via setting_sources, not inline configs"
OLD_DETAIL = "inline servers do not pick up the IDE's saved logins"
CANDIDATE = {
    "type": "howto",
    "statement": "Load Cursor OAuth MCP servers via setting_sources so saved logins are reused",
    "detail": "setting_sources=['user','project'] reads ~/.cursor/mcp.json",
    "importance": 0.6,
}
SUPERSEDE = {
    "reasoning": "the candidate replaces the old lesson",
    "statement": "Use setting_sources for every Cursor MCP server",  # the judge's replacement text
    "detail": "replaces the older inline-config advice",
}
REFUSE = {"reasoning": "adds detail, agrees", "contradicts": False, "contradiction_quote": ""}
SAME_LESSON = {"same_lesson": True, "adds_new_fact": False, "merged_statement": "",
               "merged_detail": ""}  # fmt: skip


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _supersede(store, tmp_path, **chat_kwargs):
    s, embedder = store
    path = tmp_path / "sess.jsonl"
    path.write_text(TRANSCRIPT)
    index_transcript(path, s, embedder)
    doc = s.document_id("chat", str(path.resolve()))
    target = add_learning(s, embedder, type="howto", scope=SCOPE, statement=OLD, detail=OLD_DETAIL)
    chat = FakeChat(extractions=[[dict(CANDIDATE)]],
                    decisions=[{**SUPERSEDE, "action": "supersede", "target_id": target}],
                    **chat_kwargs)  # fmt: skip
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    return s, target, chat, res


def test_supersede_without_a_real_contradiction_keeps_the_target_live(store, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="qmx.consolidate"):
        s, target, chat, res = _supersede(store, tmp_path, retire_checks=[REFUSE],
                                          dup_checks=[SAME_LESSON])
    assert (res.superseded, res.supersede_blocked, res.created, res.dedup_dropped) == (0, 1, 0, 1)
    kept = s.get_learning(target)
    assert kept.is_live and kept.superseded_by is None
    assert (kept.statement, kept.detail) == (OLD, OLD_DETAIL)
    # The candidate restates the target, so the near-duplicate check drops it: no blind duplicate.
    assert [le.learning_id for le in s.list_learnings(live_only=False)] == [target]
    # The check was shown the target (OLD) and the candidate in its own words.
    [prompt] = chat.retire_prompts
    assert f"OLD:\n{OLD}\ndetail: {OLD_DETAIL}" in prompt
    assert f"CANDIDATE:\n{CANDIDATE['statement']}" in prompt
    [dup] = chat.dup_prompts
    assert f"EXISTING:\n{OLD}" in dup
    assert f"blocked supersede of learning #{target}" in caplog.text


def test_supersede_quote_must_be_in_the_candidate(store, tmp_path):
    s, target, _, res = _supersede(store, tmp_path, retire_checks=[
        {"contradicts": True, "contradiction_quote": "inline configs work fine now"}])
    assert (res.superseded, res.supersede_blocked, res.created) == (0, 1, 1)
    assert s.get_learning(target).is_live
    # Dedup said "not the same lesson" (fake default), so it is stored — in the candidate's own
    # words, not the judge's replacement text written to stand in for the target.
    [new] = [le for le in s.list_learnings() if le.learning_id != target]
    assert (new.statement, new.detail) == (CANDIDATE["statement"], CANDIDATE["detail"])
    assert new.superseded_by is None


def test_blocked_supersede_that_adds_a_fact_is_merged_into_the_target(store, tmp_path):
    merged = OLD + " (reads ~/.cursor/mcp.json)"
    s, target, chat, res = _supersede(store, tmp_path, retire_checks=[REFUSE], dup_checks=[
        {"same_lesson": True, "adds_new_fact": True, "merged_statement": merged,
         "merged_detail": OLD_DETAIL}])  # fmt: skip
    assert (res.supersede_blocked, res.dedup_merged, res.updated, res.created) == (1, 1, 1, 0)
    kept = s.get_learning(target)
    assert kept.is_live and kept.statement == merged
    assert s.learning_revisions(target)[0].reason == "consolidate:dedup-merge"
    assert len(chat.check_prompts) == 1  # the merge still went through the preservation check


def test_verified_supersede_retires_the_target_as_before(store, tmp_path):
    s, target, chat, res = _supersede(store, tmp_path)  # default fake quotes the candidate
    assert (res.superseded, res.supersede_blocked, res.created) == (1, 0, 0)
    [new_id] = res.learning_ids
    assert s.get_learning(target).superseded_by == new_id
    assert s.get_learning(new_id).statement == SUPERSEDE["statement"]  # judge's text, as before
    assert [kind for kind, _ in chat.calls] == ["extract", "decision", "retire"]  # no dedup call


def test_unusable_retirement_check_blocks_the_supersede(store, tmp_path):
    s, target, _, res = _supersede(store, tmp_path, unusable={"retire": 1},
                                   dup_checks=[SAME_LESSON])
    assert s.get_learning(target).is_live
    assert (res.superseded, res.supersede_blocked, res.unusable_replies) == (0, 1, 1)
    assert res.dedup_dropped == 1 and len(s.list_learnings(live_only=False)) == 1


def test_supersede_check_is_a_short_capped_call(store, tmp_path):
    _, _, chat, _ = _supersede(store, tmp_path)
    assert ("retire", SHORT_REPLY_MAX_TOKENS) in chat.calls
    assert all(cap == SHORT_REPLY_MAX_TOKENS for kind, cap in chat.calls if kind != "extract")


def test_cli_reports_blocked_supersedes(tmp_path, monkeypatch, capsys):
    embedder = FakeEmbedder(dim=64)
    db = tmp_path / "index.db"
    with Store.open(db, embedder.dim, "fake") as s:
        target = add_learning(s, embedder, type="howto", scope=SCOPE, statement=OLD,
                              detail=OLD_DETAIL)
    session = tmp_path / "sess.jsonl"
    session.write_text(TRANSCRIPT)
    settings = Settings(db_path=db, embed_dim=embedder.dim, embed_model="fake",
                        ollama_url="http://127.0.0.1:9", consolidate_min_turns=0,
                        consolidate_min_chars=0)  # fmt: skip
    chat = FakeChat(extractions=[[dict(CANDIDATE)]],
                    decisions=[{**SUPERSEDE, "action": "supersede", "target_id": target}],
                    retire_checks=[REFUSE], dup_checks=[SAME_LESSON])  # fmt: skip
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, *a, **k: settings))
    monkeypatch.setattr(cli, "OllamaEmbedder", lambda _s: contextlib.nullcontext(embedder))
    monkeypatch.setattr(cli, "make_chat", lambda _s: contextlib.nullcontext(chat))
    assert cli.main(["consolidate", "--session", str(session), "--scope", SCOPE]) == 0
    out = capsys.readouterr().out
    assert "0 new, 0 updated, 0 superseded" in out
    assert "1 supersede(s) blocked by the retirement guardrail" in out
