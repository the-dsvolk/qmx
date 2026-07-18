"""Phase B/C: extract turns → learnings, dedup via new/update/supersede, idempotent re-run."""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import consolidate_session, extract_learnings
from qmx.index import index_transcript
from qmx.learnings import add_learning, lessons
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

TRANSCRIPT = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "user", "message": {"role": "user", "content": "raise IAM PRs at project level"}},
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "bucket-level IAM failed earlier"},
        },
    ]
)


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


def test_extract_filters_bad_candidates():
    chat = FakeChat(
        extractions=[
            [
                {"type": "mistake", "statement": "good one"},
                {"type": "nonsense", "statement": "bad type"},
                {"type": "howto", "statement": ""},  # empty statement
            ]
        ]
    )
    out = extract_learnings(chat, [{"role": "user", "text": "hi"}])
    assert [c["statement"] for c in out] == ["good one"]


def test_consolidate_creates_learnings(store, tmp_path):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    chat = FakeChat(
        extractions=[
            [
                {"type": "mistake", "statement": "bucket-level IAM fails; use project level",
                 "detail": "ask in #platform", "importance": 0.9},
                {"type": "howto", "statement": "run project-level IAM PRs"},
            ]
        ]
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert res.candidates == 2
    assert res.created == 2
    assert res.chunks_consolidated == 2
    found = lessons(s, embedder, "IAM project level", k=5)
    assert any("project level" in le["statement"] for le in found)
    assert found[0]["citations"], "learnings should carry source citations"


def test_consolidate_is_idempotent(store, tmp_path):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    chat = FakeChat(extractions=[[{"type": "decision", "statement": "use uv not pip"}]])
    first = consolidate_session(s, embedder, chat, doc_id)
    assert first.created == 1
    # Re-run: all turns already consolidated -> nothing read, nothing created.
    second = consolidate_session(s, embedder, chat, doc_id)
    assert second.turns_read == 0 and second.candidates == 0 and second.created == 0


def test_consolidate_supersede_replaces_stale(store, tmp_path):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    stale = add_learning(
        s, embedder, type="mistake", statement="bucket-level IAM is fine", scope="the-dsvolk/qmx"
    )
    chat = FakeChat(
        extractions=[
            [{"type": "mistake", "statement": "bucket-level IAM fails; use project level"}]
        ],
        decisions=[{"action": "supersede", "target_id": stale}],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert res.superseded == 1
    assert s.get_learning(stale).superseded_by is not None
    ids = [le["learning_id"] for le in lessons(s, embedder, "bucket-level IAM", k=5)]
    assert stale not in ids  # superseded excluded from recall


def test_consolidate_update_patches_existing(store, tmp_path):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    existing = add_learning(
        s, embedder, type="howto", statement="raise IAM PRs", scope="the-dsvolk/qmx"
    )
    chat = FakeChat(
        extractions=[[{"type": "howto", "statement": "raise IAM PRs carefully"}]],
        decisions=[
            {"action": "update", "target_id": existing, "statement": "raise IAM PRs project-level"}
        ],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert res.updated == 1 and res.created == 0
    assert s.get_learning(existing).statement == "raise IAM PRs project-level"
    # No duplicate learning created.
    assert len(s.list_learnings()) == 1
