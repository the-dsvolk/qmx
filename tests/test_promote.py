"""Phase E: promote a lesson to per-repo curated memory (isolated, deduped, inject-excluded)."""

from __future__ import annotations

import pytest

from qmx.learnings import add_learning, inject_lessons
from qmx.promote import (
    PromotionError,
    memory_dir_for,
    promotable,
    promote,
    repo_dir_name,
    slugify,
)
from qmx.store import Store
from tests.fakes import FakeEmbedder


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def test_repo_dir_name_and_slug():
    assert repo_dir_name("the-dsvolk/qmx") == "the-dsvolk__qmx"
    assert repo_dir_name(None) == "_global"
    assert slugify("Bucket-level IAM PRs fail!") == "bucket-level-iam-prs-fail"


def test_promote_writes_isolated_file_with_frontmatter(store, tmp_path):
    s, embedder = store
    lid = add_learning(
        s,
        embedder,
        type="mistake",
        statement="bucket-level IAM fails; use project level",
        detail="ask in #platform-security-support",
        scope="the-dsvolk/qmx",
        importance=0.9,
    )
    path = promote(s, lid, memory_root=tmp_path)
    # Isolated under the repo-keyed dir, not commingled.
    assert path.parent == memory_dir_for(tmp_path, "the-dsvolk/qmx")
    assert path.parent.name == "the-dsvolk__qmx"
    text = path.read_text()
    assert "name: bucket-level-iam-fails-use-project-level" in text
    assert "type: feedback" in text  # mistake -> feedback
    assert "**Why:** ask in #platform-security-support" in text
    # Pointer landed in that dir's MEMORY.md.
    index = (path.parent / "MEMORY.md").read_text()
    assert "(bucket-level-iam-fails-use-project-level.md)" in index
    # Learning marked promoted.
    assert s.get_learning(lid).promoted_to == str(path)


def test_global_lesson_goes_to_global_dir(store, tmp_path):
    s, embedder = store
    lid = add_learning(s, embedder, type="decision", statement="always branch before editing")
    path = promote(s, lid, memory_root=tmp_path)
    assert path.parent.name == "_global"
    assert "type: project" in path.read_text()  # decision -> project


def test_promoted_lesson_excluded_from_injection(store, tmp_path):
    s, embedder = store
    keep = add_learning(s, embedder, type="mistake", statement="keep injecting me", scope="me/r")
    graduate = add_learning(s, embedder, type="mistake", statement="graduated away", scope="me/r")
    promote(s, graduate, memory_root=tmp_path)
    statements = {le.statement for le in inject_lessons(s, "me/r")}
    assert "keep injecting me" in statements
    assert "graduated away" not in statements  # promoted -> not re-injected
    assert keep  # silence unused


def test_promote_dedups_by_slug(store, tmp_path):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="use uv not pip", scope="me/r")
    p1 = promote(s, lid, memory_root=tmp_path)
    # A second lesson with the same statement -> same slug -> same file (updated), no MEMORY.md dup.
    lid2 = add_learning(s, embedder, type="howto", statement="use uv not pip", scope="me/r")
    p2 = promote(s, lid2, memory_root=tmp_path)
    assert p1 == p2
    pointers = (p1.parent / "MEMORY.md").read_text().count("(use-uv-not-pip.md)")
    assert pointers == 1
    assert len(list(p1.parent.glob("*.md"))) == 2  # the lesson file + MEMORY.md, no duplicate


def test_superseded_cannot_be_promoted(store, tmp_path):
    s, embedder = store
    old = add_learning(s, embedder, type="decision", statement="old idea")
    new = add_learning(s, embedder, type="decision", statement="new idea")
    s.supersede_learning(old, new)
    with pytest.raises(PromotionError):
        promote(s, old, memory_root=tmp_path)


def test_promotable_gate(store):
    s, embedder = store
    weak = add_learning(s, embedder, type="howto", statement="weak lesson", importance=0.2)
    strong = add_learning(s, embedder, type="howto", statement="strong lesson", importance=0.9)
    s.touch_learning(strong)  # reuse_count -> 1
    ids = {le.learning_id for le in promotable(s, min_importance=0.6, min_reuse=1)}
    assert strong in ids
    assert weak not in ids
