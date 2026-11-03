"""Merge guardrails: revision history on every update, revert, and the preservation check that
stops a consolidation `update` from overwriting an unrelated lesson."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from qmx.consolidate import consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning, revert_learning, update_learning
from qmx.mcp_server import build_server
from qmx.service import QmxService
from qmx.store import SCHEMA_VERSION, Store
from tests.fakes import FakeChat, FakeEmbedder, build_index

TRANSCRIPT = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "user", "message": {"role": "user", "content": "check wash sales everywhere"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "store lots per acct"}},
    ]
)
SCOPE = "the-dsvolk/qmx"


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _chat_doc(store, embedder, tmp_path):
    path = tmp_path / "sess.jsonl"
    path.write_text(TRANSCRIPT)
    index_transcript(path, store, embedder)
    return store.document_id("chat", str(path.resolve()))


# -- revision history ------------------------------------------------------------------------


def test_every_update_snapshots_the_previous_version(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="v1", detail="d1")
    update_learning(s, embedder, lid, statement="v2")
    update_learning(s, embedder, lid, importance=0.9)
    revs = s.learning_revisions(lid)
    assert [r.statement for r in revs] == ["v2", "v1"]  # newest first
    assert revs[-1].detail == "d1"
    assert {r.reason for r in revs} == {"manual"}


def test_schema_v5_store_is_upgraded_in_place(tmp_path):
    db = tmp_path / "index.db"
    embedder = FakeEmbedder(dim=64)
    with Store.open(db, embedder.dim, "fake") as s:
        lid = add_learning(s, embedder, type="howto", statement="kept across upgrade")
    conn = sqlite3.connect(db)  # simulate a store written by the previous release
    conn.execute("DROP TABLE learning_revisions")
    conn.execute("PRAGMA user_version=5")
    conn.commit()
    conn.close()
    with Store.open(db, embedder.dim, "fake") as s:
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 6  # noqa: SLF001
        assert s.get_learning(lid).statement == "kept across upgrade"
        update_learning(s, embedder, lid, statement="edited after upgrade")
        assert [r.statement for r in s.learning_revisions(lid)] == ["kept across upgrade"]


def test_revert_restores_latest_and_is_itself_reversible(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="original", detail="keep me")
    update_learning(s, embedder, lid, statement="clobbered", detail="other subject")
    reverted = revert_learning(s, embedder, lid)
    assert (reverted.statement, reverted.detail) == ("original", "keep me")
    assert s.learning_revisions(lid)[0].reason.startswith("revert to r")
    # Reverting again undoes the revert.
    assert revert_learning(s, embedder, lid).statement == "clobbered"


def test_revert_to_a_specific_revision_and_errors(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="a")
    other = add_learning(s, embedder, type="howto", statement="b")
    with pytest.raises(ValueError, match="no revisions"):
        revert_learning(s, embedder, lid)
    update_learning(s, embedder, lid, statement="a2")
    update_learning(s, embedder, lid, statement="a3")
    oldest = s.learning_revisions(lid)[-1]
    assert revert_learning(s, embedder, lid, oldest.revision_id).statement == "a"
    update_learning(s, embedder, other, statement="b2")
    foreign = s.learning_revisions(other)[0].revision_id
    with pytest.raises(ValueError, match="not a revision"):
        revert_learning(s, embedder, lid, foreign)
    assert revert_learning(s, embedder, 9999) is None


# -- preservation check on consolidation updates ---------------------------------------------


def _update_case(store, tmp_path, checks):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    target = add_learning(
        s, embedder, type="howto", scope=SCOPE,
        statement="Track wash-sale freezes across all household accounts",
        detail="the IRS treats spouse, IRAs and RSUs as one entity",
    )  # fmt: skip
    chat = FakeChat(
        extractions=[[{"type": "decision", "statement": "Store lots and trades in separate tables",
                       "detail": "lots are snapshots, trades are events", "importance": 0.6}]],
        decisions=[{"reasoning": "merge", "action": "update", "target_id": target,
                    "statement": "Separate lots and trades into distinct tables",
                    "detail": "schema for wash-sale logic"}],
        checks=checks,
    )  # fmt: skip
    res = consolidate_session(s, embedder, chat, doc_id, scope=SCOPE)
    return s, target, chat, res


def test_update_that_loses_the_targets_claim_is_blocked(store, tmp_path):
    s, target, chat, res = _update_case(
        store, tmp_path, [{"preserves": False, "lost": "household wash-sale scope"}]
    )
    assert (res.updated, res.update_blocked, res.created) == (0, 1, 1)
    kept = s.get_learning(target)
    assert kept.statement == "Track wash-sale freezes across all household accounts"
    assert s.learning_revisions(target) == []  # untouched, nothing to revert
    [new] = [le for le in s.list_learnings() if le.learning_id != target]
    # Stored in the candidate's own words, not the judge's blended merge text.
    assert new.statement == "Store lots and trades in separate tables"
    assert new.detail == "lots are snapshots, trades are events"
    # The check was shown the target (OLD) and the proposed merge (NEW).
    [prompt] = chat.check_prompts
    assert "OLD:\nTrack wash-sale freezes" in prompt
    assert "NEW:\nSeparate lots and trades" in prompt


def test_update_that_preserves_the_target_is_applied_and_revisioned(store, tmp_path):
    s, target, _, res = _update_case(store, tmp_path, [{"preserves": True, "lost": ""}])
    assert (res.updated, res.update_blocked, res.created) == (1, 0, 0)
    assert s.get_learning(target).statement == "Separate lots and trades into distinct tables"
    [rev] = s.learning_revisions(target)
    assert rev.reason == "consolidate:update"
    assert rev.statement == "Track wash-sale freezes across all household accounts"


def test_unusable_check_reply_fails_safe(store, tmp_path):
    s, target, _, res = _update_case(store, tmp_path, [{}])  # garbage / empty reply
    assert (res.updated, res.update_blocked) == (0, 1)
    assert s.get_learning(target).statement.startswith("Track wash-sale freezes")


# -- agent access ----------------------------------------------------------------------------


def test_history_and_revert_are_mcp_tools(tmp_path):
    embedder = FakeEmbedder(dim=64)
    settings = build_index(tmp_path, embedder, {"a.py": "x = 1\n"})
    server = build_server(settings, QmxService(settings, embedder))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"learning_history", "revert_learning"} <= names

    svc = QmxService(settings, embedder)
    lid = svc.add_learning(statement="first wording", type="howto")["learning_id"]
    svc.update_learning(lid, statement="second wording")
    [rev] = svc.learning_history(lid)
    assert rev["statement"] == "first wording" and rev["reason"] == "manual"
    assert svc.revert_learning(lid)["statement"] == "first wording"
    assert svc.learning_history(424242) is None
