"""Insert-time near-duplicate check and similarity/metadata in the judge's neighbour listing."""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

SCOPE = "the-dsvolk/qmx"
EXISTING = "Store API secrets in a restricted local directory such as secret project folder"
CANDIDATE = "Store API secrets in a restricted local directory such as the secret project folder"


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _run(store, tmp_path, chat, **kw):
    s, embedder = store
    path = tmp_path / "sess.jsonl"
    path.write_text("\n".join(json.dumps({"type": r, "message": {"role": r, "content": t}})
                              for r, t in (("user", "where do secrets go"),
                                           ("assistant", "a restricted local folder"))))
    index_transcript(path, s, embedder)
    doc = s.document_id("chat", str(path.resolve()))
    existing = add_learning(s, embedder, type="howto", statement=EXISTING, scope=SCOPE)
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE, **kw)
    return s, existing, res


def _chat(**kw):
    return FakeChat(extractions=[[{"type": "howto", "statement": CANDIDATE,
                                   "detail": "chmod 700 the folder"}]], **kw)


def test_restatement_is_dropped(store, tmp_path):
    chat = _chat(dup_checks=[{"same_lesson": True, "adds_new_fact": False,
                              "merged_statement": "", "merged_detail": ""}])
    s, existing, res = _run(store, tmp_path, chat)
    assert (res.created, res.dedup_dropped, res.dedup_merged) == (0, 1, 0)
    assert len(s.list_learnings()) == 1 and s.learning_revisions(existing) == []
    assert "EXISTING:\n" + EXISTING in chat.dup_prompts[0]


def test_new_fact_is_merged_with_a_revision(store, tmp_path):
    merged = EXISTING + ", with chmod 700"
    chat = _chat(dup_checks=[{"same_lesson": True, "adds_new_fact": True,
                              "merged_statement": merged, "merged_detail": "chmod 700 the folder"}])
    s, existing, res = _run(store, tmp_path, chat)
    assert (res.created, res.updated, res.dedup_merged) == (0, 1, 1)
    assert s.get_learning(existing).statement == merged
    [rev] = s.learning_revisions(existing)
    assert rev.reason == "consolidate:dedup-merge" and rev.statement == EXISTING


def test_merge_that_loses_the_claim_drops_the_candidate(store, tmp_path):
    chat = _chat(
        dup_checks=[{"same_lesson": True, "adds_new_fact": True,
                     "merged_statement": "Use chmod 700", "merged_detail": ""}],
        checks=[{"preserves": False, "evidence_in_new": "", "unsupported_additions": [],
                 "lost": "where secrets live"}],
    )
    s, existing, res = _run(store, tmp_path, chat)
    assert (res.created, res.updated, res.dedup_dropped) == (0, 0, 1)
    assert s.get_learning(existing).statement == EXISTING


@pytest.mark.parametrize("check", [{"same_lesson": False}, {}], ids=["not-same", "unusable"])
def test_genuinely_new_or_unusable_reply_is_stored(store, tmp_path, check):
    _, _, res = _run(store, tmp_path, _chat(dup_checks=[check]))
    assert (res.created, res.dedup_dropped) == (1, 0)


def test_threshold_zero_disables_the_check(store, tmp_path):
    chat = _chat()
    _, _, res = _run(store, tmp_path, chat, dedup_min_similarity=0.0)
    assert res.created == 1 and chat.dup_prompts == []


def test_refused_update_falls_back_to_new_without_a_dedup_question(store, tmp_path):
    s, embedder = store
    chat = _chat(checks=[{"preserves": False, "evidence_in_new": "", "unsupported_additions": [],
                          "lost": "x"}])
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}))
    index_transcript(path, s, embedder)
    doc = s.document_id("chat", str(path.resolve()))
    target = add_learning(s, embedder, type="howto", statement=EXISTING, scope=SCOPE)
    chat._decisions = [{"action": "update", "target_id": target, "statement": "unrelated text"}]
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE)
    assert (res.update_blocked, res.created) == (1, 1) and chat.dup_prompts == []


def test_neighbour_listing_shows_similarity_and_provenance(store, tmp_path):
    chat = _chat()
    _run(store, tmp_path, chat)
    [prompt] = chat.decision_prompts
    assert "similarity " in prompt and "created " in prompt
