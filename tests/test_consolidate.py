"""Phase B/C: extract turns → learnings, dedup via new/update/supersede, idempotent re-run."""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import consolidate_session, extract_learnings
from qmx.index import index_transcript
from qmx.learnings import add_learning, deprecate_learning, lessons
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


def test_consolidate_deprecate_retires_without_storing_a_replacement(store, tmp_path):
    """The action `supersede` can't express: the lesson is wrong, nothing replaces it."""
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    wrong = add_learning(
        s, embedder, type="mistake", statement="bucket-level IAM is fine", scope="the-dsvolk/qmx"
    )
    chat = FakeChat(
        extractions=[[{"type": "mistake", "statement": "bucket-level IAM never worked at all"}]],
        decisions=[
            {"action": "deprecate", "target_id": wrong, "reason": "never true; no rule replaces it"}
        ],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert (res.deprecated, res.created, res.superseded) == (1, 0, 0)
    retired = s.get_learning(wrong)
    assert retired.is_deprecated and retired.deprecated_reason == "never true; no rule replaces it"
    assert retired.superseded_by is None  # retired, not replaced
    assert len(s.list_learnings(live_only=False)) == 1, "deprecate must not insert the candidate"
    assert lessons(s, embedder, "bucket-level IAM", k=5) == []


def test_consolidate_shows_retired_lessons_to_the_judge_and_can_drop(store, tmp_path):
    """The regression this closes: a retired lesson is invisible to search, so without
    ``include_retired`` the judge sees no match and re-adds it as `new` next session."""
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    retired = add_learning(
        s, embedder, type="mistake", statement="bucket-level IAM is fine", scope="the-dsvolk/qmx"
    )
    deprecate_learning(s, retired, reason="wrong; project level is required")

    chat = FakeChat(
        extractions=[[{"type": "mistake", "statement": "bucket-level IAM is fine, use it"}]],
        decisions=[{"action": "drop"}],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")

    prompt = chat.decision_prompts[0]
    assert f"id={retired}" in prompt, "the judge must be shown the retired lesson"
    assert "[RETIRED: wrong; project level is required]" in prompt, "...flagged, with the reason"
    assert res.dropped == 1 and res.created == 0
    assert len(s.list_learnings(live_only=False)) == 1, "no new row: the retirement holds"
    assert s.get_learning(retired).is_deprecated  # and it stays retired


def test_consolidate_will_not_edit_or_re_retire_a_retired_lesson(store, tmp_path):
    """A retired target is shown for context only — editing it would undo the retirement."""
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    retired = add_learning(
        s, embedder, type="mistake", statement="bucket-level IAM is fine", scope="the-dsvolk/qmx"
    )
    deprecate_learning(s, retired, reason="wrong")
    chat = FakeChat(
        extractions=[[{"type": "mistake", "statement": "bucket-level IAM fails; project level"}]],
        decisions=[{"action": "update", "target_id": retired, "statement": "hijacked"}],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert res.updated == 0 and res.created == 1  # falls through to insert, candidate not lost
    assert s.get_learning(retired).statement == "bucket-level IAM is fine"  # untouched
    assert s.get_learning(retired).is_deprecated


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
