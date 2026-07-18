"""Phase A acceptance: seed lessons, retrieve by relevance×importance×recency, with citations."""

from __future__ import annotations

import pytest

from qmx.learnings import add_learning, inject_lessons, lessons
from qmx.store import Store
from tests.fakes import FakeEmbedder


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def test_add_and_recall_lesson_with_citation(store):
    s, embedder = store
    lid = add_learning(
        s,
        embedder,
        type="mistake",
        statement="bucket-level IAM PRs fail; raise them at project level",
        detail="ask in #platform-security-support",
        scope="the-dsvolk/qmx",
        importance=0.9,
        source_anchors=[{"session": "abc", "line": 42}],
    )
    assert lid == 1
    results = lessons(s, embedder, "iam pr project level", k=5)
    assert results, "expected the seeded lesson back"
    top = results[0]
    assert top["learning_id"] == lid
    assert top["type"] == "mistake"
    assert top["citations"] == [{"session": "abc", "line": 42}]
    assert "score" in top


def test_ranking_blends_importance(store):
    s, embedder = store
    # Two lessons with the same content words -> same relevance; importance breaks the tie.
    low = add_learning(
        s, embedder, type="howto", statement="deploy the service with care", importance=0.1
    )
    high = add_learning(
        s, embedder, type="howto", statement="deploy the service with care", importance=0.95
    )
    results = lessons(s, embedder, "deploy the service", k=2)
    ids = [r["learning_id"] for r in results]
    assert ids[0] == high and ids[1] == low, f"importance should rank {high} first, got {ids}"


def test_superseded_excluded_from_recall(store):
    s, embedder = store
    old = add_learning(s, embedder, type="decision", statement="use tabs for indentation")
    new = add_learning(s, embedder, type="decision", statement="use tabs for indentation always")
    s.supersede_learning(old, new)
    ids = [r["learning_id"] for r in lessons(s, embedder, "tabs indentation", k=5)]
    assert old not in ids
    assert new in ids


def test_type_filter(store):
    s, embedder = store
    add_learning(s, embedder, type="mistake", statement="alpha vector search failed once")
    add_learning(s, embedder, type="howto", statement="alpha vector search works this way")
    only_mistakes = lessons(s, embedder, "alpha vector search", k=5, type="mistake")
    assert only_mistakes and all(r["type"] == "mistake" for r in only_mistakes)


def test_reuse_count_increments_on_recall(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="unique gamma retrieval trick")
    assert s.get_learning(lid).reuse_count == 0
    lessons(s, embedder, "gamma retrieval", k=1)
    assert s.get_learning(lid).reuse_count == 1


def test_scope_filter_and_global(store):
    s, embedder = store
    add_learning(s, embedder, type="howto", statement="repo delta helper", scope="me/repoA")
    add_learning(s, embedder, type="howto", statement="repo delta helper", scope="me/repoB")
    add_learning(s, embedder, type="howto", statement="repo delta helper global", scope=None)
    scoped = lessons(s, embedder, "repo delta helper", k=5, scope="me/repoA")
    scopes = {r["scope"] for r in scoped}
    assert "me/repoB" not in scopes  # other repo excluded
    assert scopes <= {"me/repoA", None}  # own repo + global only


def test_inject_lessons_is_query_free_and_scope_keyed(store):
    s, embedder = store
    add_learning(s, embedder, type="mistake", statement="A repo lesson", scope="me/repoA")
    add_learning(s, embedder, type="mistake", statement="B repo lesson", scope="me/repoB")
    add_learning(s, embedder, type="decision", statement="global lesson", scope=None)
    injected = inject_lessons(s, "me/repoA")
    statements = {le.statement for le in injected}
    assert "A repo lesson" in statements
    assert "global lesson" in statements  # globals always injected
    assert "B repo lesson" not in statements  # other repo never leaks in


def test_inject_respects_char_budget(store):
    s, embedder = store
    for i in range(10):
        add_learning(
            s, embedder, type="howto", statement=f"lesson number {i} " + "x" * 100, scope="me/r"
        )
    injected = inject_lessons(s, "me/r", char_budget=250)
    assert 0 < len(injected) < 10  # budget caps how many fit
