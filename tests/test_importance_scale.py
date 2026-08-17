"""`importance` must stay on the documented 0..1 scale — it is a ranking weight, not a rating.

`lessons()` blends `W_RELEVANCE * relevance + W_IMPORTANCE * importance + W_RECENCY * recency`. The
relevance term is capped at `W_RELEVANCE` (0.5), so a lesson stored at 9.0 contributes 2.7 and
outranks every genuinely relevant lesson — and clears the `promotable(min_importance=…)` gate for
free. A real index had 281/1146 rows above 1.0 because consolidation's `update` decision wrote the
judge's raw number straight through `store.update_learning`.
"""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import consolidate_session
from qmx.index import index_transcript
from qmx.learnings import add_learning, lessons, normalize_importance
from qmx.promote import promotable
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (4.0, 0.8),  # 1-5 rating
        (5.0, 1.0),
        (9.0, 0.9),  # 1-10 rating
        (10.0, 1.0),
        (-2.0, 0.0),  # clamped, not reflected
        (100.0, 1.0),
    ],
)
def test_normalize_importance_rescales_by_apparent_scale(raw, expected):
    assert normalize_importance(raw) == pytest.approx(expected)


@pytest.mark.parametrize("junk", [None, "high", "", object()])
def test_normalize_importance_falls_back_to_the_default_on_junk(junk):
    """It sits on a model-output path, so unparseable input must not raise."""
    assert normalize_importance(junk) == 0.5


def test_normalize_importance_preserves_relative_order(store):
    """Rescaling, not clamping: clamping would flatten 4.0 and 9.0 to the same top priority."""
    assert normalize_importance(2.0) < normalize_importance(4.0) < normalize_importance(9.0)


def test_store_clamps_importance_on_insert_and_update(store):
    s, _embedder = store
    lid = s.insert_learning(type="howto", statement="clamped on insert", importance=7.0)
    assert s.get_learning(lid).importance == 1.0
    s.update_learning(lid, importance=4.0)
    assert s.get_learning(lid).importance == 1.0
    s.update_learning(lid, importance=-1.0)
    assert s.get_learning(lid).importance == 0.0


def test_consolidate_update_normalizes_the_judges_rating(store, tmp_path):
    """The actual regression: `update` used to write the judge's 1-5 answer verbatim."""
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    existing = add_learning(
        s, embedder, type="howto", statement="raise IAM PRs", scope="the-dsvolk/qmx"
    )
    chat = FakeChat(
        extractions=[[{"type": "howto", "statement": "raise IAM PRs carefully"}]],
        decisions=[{"action": "update", "target_id": existing, "importance": 4}],
    )
    res = consolidate_session(s, embedder, chat, doc_id, scope="the-dsvolk/qmx")
    assert res.updated == 1
    assert s.get_learning(existing).importance == pytest.approx(0.8)


def test_consolidate_new_normalizes_the_judges_rating(store, tmp_path):
    s, embedder = store
    doc_id = _chat_doc(s, embedder, tmp_path)
    chat = FakeChat(extractions=[[{"type": "decision", "statement": "use uv", "importance": 9}]])
    consolidate_session(s, embedder, chat, doc_id)
    [learning] = s.list_learnings(live_only=False)
    assert learning.importance == pytest.approx(0.9)


def test_out_of_range_importance_would_dominate_ranking(store):
    """Why this matters: the mis-scaled lesson wins even when the other is a better match."""
    s, embedder = store
    relevant = add_learning(
        s, embedder, type="howto", statement="alpha beta gamma delta retrieval", importance=0.9
    )
    off_topic = add_learning(s, embedder, type="howto", statement="totally unrelated epsilon")
    s.set_learning_importance(off_topic, 0.5)
    assert [h["learning_id"] for h in lessons(s, embedder, "alpha beta gamma delta", k=2)][0] == (
        relevant
    ), "sanity: with sane weights the relevant lesson ranks first"

    s._conn.execute(  # noqa: SLF001 - bypass the clamp to reproduce the legacy bad data
        "UPDATE learnings SET importance=9.0 WHERE learning_id=?", (off_topic,)
    )
    s._conn.commit()  # noqa: SLF001
    top = lessons(s, embedder, "alpha beta gamma delta", k=2)[0]
    assert top["learning_id"] == off_topic, "an unscaled weight buries the relevant lesson"


def test_out_of_range_importance_clears_the_promotion_gate(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="weak lesson", importance=0.1)
    s._conn.execute(  # noqa: SLF001 - legacy bad data
        "UPDATE learnings SET importance=5.0, reuse_count=1 WHERE learning_id=?", (lid,)
    )
    s._conn.commit()  # noqa: SLF001
    assert [le.learning_id for le in promotable(s, min_importance=0.6, min_reuse=1)] == [lid]
    # ...and the repair drops it back below the gate.
    s.set_learning_importance(lid, normalize_importance(5.0) * 0.1)
    assert promotable(s, min_importance=0.6, min_reuse=1) == []


def test_set_learning_importance_does_not_bump_updated_at(store):
    """A bulk repair must not make every fixed lesson look freshly learned (recency term)."""
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="a lesson", importance=0.5)
    before = s.get_learning(lid).updated_at
    s.set_learning_importance(lid, 0.2)
    after = s.get_learning(lid)
    assert after.importance == pytest.approx(0.2)
    assert after.updated_at == before
