"""Lifecycle ops: fix a lesson in place, soft-retire it (with or without a replacement), restore.

The properties that matter operationally:

- an update is *in place* — same ``learning_id``, no near-duplicate row, and no embedding call when
  only ``importance``/``scope`` moved (so re-weighting works with the backend down);
- a retired lesson disappears from **every** read path, not just ``lessons()`` — that is the
  "invisible by default" contract — while staying inspectable with a breadcrumb on request.
"""

from __future__ import annotations

import pytest

from qmx.learnings import (
    add_learning,
    deprecate_learning,
    inject_lessons,
    lessons,
    restore_learning,
    update_learning,
)
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    """A :class:`FakeEmbedder` that records how many texts it was asked to embed."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed(texts)


@pytest.fixture
def store():
    embedder = CountingEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


# -- update ------------------------------------------------------------------------------------


def test_update_fixes_statement_in_place_and_is_retrievable_by_new_wording(store):
    s, embedder = store
    lid = add_learning(
        s, embedder, type="mistake", statement="H100 on-demand costs 3.08 per hour", importance=0.8
    )
    updated = update_learning(
        s, embedder, lid, statement="H100 on-demand costs 2.0725 per hour", importance=0.6
    )
    assert updated.learning_id == lid  # same row, not a near-duplicate
    assert updated.importance == pytest.approx(0.6)
    hits = lessons(s, embedder, "H100 on-demand hourly price", k=5)
    assert [h["learning_id"] for h in hits] == [lid]
    assert "2.0725" in hits[0]["statement"]


def test_importance_only_update_skips_the_embedding_call(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="epsilon caching trick")
    before = embedder.calls
    update_learning(s, embedder, lid, importance=0.05)
    assert embedder.calls == before, "re-weighting must not hit the embedding backend"
    # ...while a statement change does re-embed, so retrieval reflects the new wording.
    update_learning(s, embedder, lid, statement="epsilon caching trick, revised")
    assert embedder.calls > before


def test_update_can_fix_type_and_topic_and_clear_a_nullable_field(store):
    s, embedder = store
    lid = add_learning(
        s, embedder, type="howto", statement="zeta rollout steps", topic="rollout", detail="why"
    )
    updated = update_learning(s, embedder, lid, type="decision", topic=None, detail=None)
    assert updated.type == "decision"
    assert updated.topic is None and updated.detail is None  # None clears; KEEP would leave alone


def test_rescoping_updates_the_learning_document_repo(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="eta scoped lesson", scope="me/repoA")
    update_learning(s, embedder, lid, scope="me/repoB")
    doc_repo = s._conn.execute(  # noqa: SLF001 - asserting the mirrored column directly
        "SELECT repo FROM documents WHERE doc_id=?", (s.get_learning(lid).doc_id,)
    ).fetchone()[0]
    assert doc_repo == "me/repoB", "documents.repo must follow scope, not drift"
    update_learning(s, embedder, lid, scope=None)
    assert s.get_learning(lid).scope is None


def test_update_rejects_a_bad_type_and_an_empty_statement(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="theta lesson")
    with pytest.raises(ValueError):
        update_learning(s, embedder, lid, type="opinion")
    with pytest.raises(ValueError):
        update_learning(s, embedder, lid, statement="   ")
    assert s.get_learning(lid).type == "howto"  # nothing written on rejection


def test_update_of_a_missing_learning_returns_none(store):
    s, embedder = store
    assert update_learning(s, embedder, 999, importance=0.1) is None


# -- deprecate / restore -----------------------------------------------------------------------


def test_deprecate_without_a_replacement_hides_the_lesson(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="decision", statement="iota approach is the way")
    retired = deprecate_learning(s, lid, reason="turned out to be wrong")
    assert retired.is_deprecated and not retired.is_live
    assert retired.superseded_by is None  # nothing replaces it — the case supersede can't express
    assert lessons(s, embedder, "iota approach", k=5) == []


def test_deprecate_with_a_replacement_keeps_the_breadcrumb(store):
    s, embedder = store
    old = add_learning(s, embedder, type="mistake", statement="kappa rate is 3.08")
    new = add_learning(s, embedder, type="mistake", statement="kappa rate is 2.0725")
    deprecate_learning(s, old, reason="renegotiated", superseded_by=new)

    live = [h["learning_id"] for h in lessons(s, embedder, "kappa rate", k=5)]
    assert live == [new]

    both = lessons(s, embedder, "kappa rate", k=5, include_retired=True)
    with_retired = {h["learning_id"]: h for h in both}
    assert set(with_retired) == {old, new}
    assert with_retired[old]["superseded_by"] == new
    assert with_retired[old]["deprecated_reason"] == "renegotiated"


def test_retired_lesson_is_hidden_from_raw_query_too(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="lambda unique retrieval marker")
    assert search(s, embedder, "lambda unique retrieval marker", k=5, kind="learning")
    deprecate_learning(s, lid, reason="stale")
    # The leak this closes: `query --kind learning` bypasses lessons() entirely.
    assert search(s, embedder, "lambda unique retrieval marker", k=5, kind="learning") == []
    assert search(s, embedder, "lambda unique retrieval marker", k=5) == []
    assert search(
        s, embedder, "lambda unique retrieval marker", k=5, kind="learning", include_retired=True
    )


def test_superseded_lesson_is_hidden_from_raw_query_too(store):
    """The pre-existing leak: supersede was only filtered inside ``lessons()``."""
    s, embedder = store
    old = add_learning(s, embedder, type="decision", statement="mu unique legacy marker")
    new = add_learning(s, embedder, type="decision", statement="mu unique current marker")
    s.supersede_learning(old, new)

    def ids(**kw):
        hits = search(s, embedder, "mu unique legacy marker", k=5, kind="learning", **kw)
        return {s.learning_by_doc_id(h.hit.doc_id).learning_id for h in hits}

    assert ids() == {new}  # the replacement is still found, the stale row is not
    assert ids(include_retired=True) == {old, new}


def test_retired_lesson_is_not_injected_and_earns_no_reuse_credit(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="mistake", statement="nu lesson", scope="me/r")
    deprecate_learning(s, lid, reason="wrong")
    assert [le.learning_id for le in inject_lessons(s, "me/r")] == []
    lessons(s, embedder, "nu lesson", k=5, include_retired=True)
    assert s.get_learning(lid).reuse_count == 0, "inspection must not count toward promotion"


def test_deprecate_is_idempotent_and_keeps_the_original_timestamp(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="xi lesson")
    first = deprecate_learning(s, lid, reason="first reason")
    again = deprecate_learning(s, lid)
    assert again.deprecated_at == first.deprecated_at
    assert again.deprecated_reason == "first reason", "a bare re-deprecate keeps the reason"


def test_restore_revives_a_retired_lesson_without_re_embedding(store):
    s, embedder = store
    old = add_learning(s, embedder, type="decision", statement="omicron rule applies")
    new = add_learning(s, embedder, type="decision", statement="omicron rule replacement")
    deprecate_learning(s, old, reason="oops", superseded_by=new)
    before = embedder.calls

    revived = restore_learning(s, old)
    assert revived.is_live and revived.deprecated_at is None and revived.superseded_by is None
    assert embedder.calls == before, "restore is metadata-only — the embedding was never discarded"
    assert old in {h["learning_id"] for h in lessons(s, embedder, "omicron rule", k=5)}


def test_deprecate_rejects_a_self_or_unknown_replacement(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="pi lesson")
    with pytest.raises(ValueError):
        deprecate_learning(s, lid, superseded_by=lid)
    with pytest.raises(ValueError):
        deprecate_learning(s, lid, superseded_by=4242)
    assert s.get_learning(lid).is_live


def test_deprecate_and_restore_of_a_missing_learning_return_none(store):
    s, _embedder = store
    assert deprecate_learning(s, 999) is None
    assert restore_learning(s, 999) is None


def test_v4_database_migrates_in_place_and_keeps_its_learnings(tmp_path):
    """A v4 index (learnings, no deprecation columns) must upgrade without a rebuild."""
    embedder = CountingEmbedder(dim=64)
    db = tmp_path / "qmx.db"
    with Store.open(db, embedder.dim, "fake") as s:
        lid = add_learning(s, embedder, type="howto", statement="sigma pre-migration lesson")
        s._conn.executescript(  # noqa: SLF001 - rewind the schema to v4 on purpose
            """
            ALTER TABLE learnings DROP COLUMN deprecated_at;
            ALTER TABLE learnings DROP COLUMN deprecated_reason;
            PRAGMA user_version=4;
            """
        )
        s._conn.commit()  # noqa: SLF001

    with Store.open(db, embedder.dim, "fake") as s:
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] == 5  # noqa: SLF001
        assert s.get_learning(lid).statement == "sigma pre-migration lesson"
        assert s.get_learning(lid).is_live  # NULL deprecated_at reads as live
        assert deprecate_learning(s, lid, reason="post-migration").is_deprecated
        assert lessons(s, embedder, "sigma pre-migration", k=5) == []


def test_list_deprecated_only(store):
    s, embedder = store
    kept = add_learning(s, embedder, type="howto", statement="rho kept lesson")
    gone = add_learning(s, embedder, type="howto", statement="rho retired lesson")
    deprecate_learning(s, gone, reason="stale")
    retired = s.list_learnings(deprecated_only=True)
    assert [le.learning_id for le in retired] == [gone]
    assert kept in {le.learning_id for le in s.list_learnings(live_only=True)}
