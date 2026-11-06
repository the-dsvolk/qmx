"""A kind filter ranks within that kind, so a small kind is never crowded out by a large one.

Regression (eval, 2026-11-05): ``search_vec``/``search_fts`` applied ``kind`` *after* a global
top-``k*4``. Lessons are ~2% of chunks and the chats that taught them discuss the same topic, so the
consolidation judge was shown no existing lesson for 22% of the candidates it filed (and 1-3 for
48%) and could only file near-duplicates. Same for ``query --kind learning`` and parked items.
"""

from __future__ import annotations

import pytest

from qmx import consolidate as C
from qmx.index import reindex
from qmx.learnings import add_learning
from qmx.store import Chunk, Store
from tests.fakes import FakeEmbedder

QUERY = "household wash sale accounts"
N_CHATS = 120  # far more than any k*4 over-fetch below


@pytest.fixture
def crowded(tmp_path):
    """120 chat chunks that each share all four query words (closer than any lesson), plus the
    lessons under test, which share only three."""
    embedder = FakeEmbedder(dim=256)
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as s:
        chat = s.upsert_document(kind="chat", path="session.jsonl")
        reindex(s, embedder, chat, [Chunk(text=f"{QUERY} chat{i:03d}", ord=i)
                                    for i in range(N_CHATS)])
        yield s, embedder


def _add_lesson_chunk(s: Store, embedder: FakeEmbedder, path: str, text: str) -> None:
    doc = s.upsert_document(kind="learning", path=path)
    reindex(s, embedder, doc, [Chunk(text=text)])


def test_vector_kind_filter_finds_a_crowded_out_lesson(crowded):
    s, embedder = crowded
    _add_lesson_chunk(s, embedder, "learning:1", "household wash sale lesson rule")
    [qvec] = embedder.embed([QUERY])
    # Precondition: in the global ranking every chat outranks the lesson.
    assert all(h.kind == "chat" for h in s.search_vec(qvec, k=N_CHATS))
    hits = s.search_vec(qvec, k=5, kind="learning")
    assert [h.text for h in hits] == ["household wash sale lesson rule"]


def test_vector_kind_filter_is_an_exact_top_k(crowded):
    s, embedder = crowded
    texts = ["household wash sale lesson", "household wash lesson rule extra",
             "household lesson rule extra words"]  # 3, 2 and 1 shared query words
    for i, t in enumerate(texts):
        _add_lesson_chunk(s, embedder, f"learning:{i}", t)
    [qvec] = embedder.embed([QUERY])
    hits = s.search_vec(qvec, k=2, kind="learning")
    assert [h.text for h in hits] == texts[:2]
    assert hits[0].distance <= hits[1].distance


def test_vector_kind_filter_unchanged_when_the_kind_fills_k(crowded):
    s, embedder = crowded
    [qvec] = embedder.embed([QUERY])
    hits = s.search_vec(qvec, k=5, kind="chat")
    assert len(hits) == 5 and all(h.kind == "chat" for h in hits)
    assert hits == s.search_vec(qvec, k=5)  # chats are the global top-5 anyway


def test_vector_kind_filter_on_an_empty_kind(crowded):
    s, embedder = crowded
    [qvec] = embedder.embed([QUERY])
    assert s.search_vec(qvec, k=5, kind="meeting") == []


def test_fts_kind_filter_finds_a_crowded_out_lesson(crowded):
    s, embedder = crowded
    _add_lesson_chunk(s, embedder, "learning:1", "household wash sale lesson rule")
    hits = s.search_fts("household wash sale", k=5, kind="learning")
    assert [h.text for h in hits] == ["household wash sale lesson rule"]


def test_judge_neighbours_include_a_lesson_crowded_out_by_its_own_chat(crowded):
    """The consolidation path: the judge must be shown the existing lesson."""
    s, embedder = crowded
    lid = add_learning(s, embedder, type="howto", statement="Track household wash sale freezes",
                       detail="the IRS treats the household as one taxpayer")
    matches, sims = C._nearest_learnings(s, embedder, "household wash sale accounts rule", None)
    assert lid in [m.learning_id for m in matches]
    assert sims[lid] > 0
