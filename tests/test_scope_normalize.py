"""Hand-typed scopes are normalized to the keys session hooks use (owner/repo, or None = global)."""

from __future__ import annotations

import pytest

from qmx.learnings import add_learning, update_learning
from qmx.scope import UnknownScopeError, normalize_scope
from qmx.service import QmxService
from qmx.store import Store
from tests.fakes import FakeEmbedder, build_index

KNOWN = ["me/qmx", "work/data", "Org/Tools", "other/data", "localonly"]


@pytest.mark.parametrize("raw", [None, "", "  ", "global", "GLOBAL", "_global", "none", "null"])
def test_global_spellings_mean_global(raw):
    assert normalize_scope(raw, KNOWN, strict=True) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("me/qmx", "me/qmx"),  # already canonical
        ("ME/QMX", "me/qmx"),  # letter case follows the stored key
        ("org/tools", "Org/Tools"),
        ("qmx", "me/qmx"),  # bare repo name -> its one owner/repo
        ("tools", "Org/Tools"),
        ("git@github.com:me/qmx.git", "me/qmx"),  # a remote URL
        ("https://github.com/me/qmx", "me/qmx"),
        ("someone/new-repo", "someone/new-repo"),  # owner/repo with no lessons yet
        ("localonly", "localonly"),  # a repo with no remote, already in use
    ],
)
def test_normalize_scope_maps_to_the_canonical_key(raw, expected):
    assert normalize_scope(raw, KNOWN, strict=True) == expected


def test_bare_name_matching_several_repos_is_ambiguous():
    with pytest.raises(UnknownScopeError, match="ambiguous.*other/data.*work/data"):
        normalize_scope("data", KNOWN, strict=True)
    assert normalize_scope("data", KNOWN) == "data"  # lenient: kept as given


def test_unknown_bare_name_is_rejected_only_when_strict():
    with pytest.raises(UnknownScopeError, match="owner/repo"):
        normalize_scope("nvx-like", KNOWN, strict=True)
    assert normalize_scope("nvx-like", KNOWN) == "nvx-like"


def test_owner_match_wins_over_a_stray_bare_key():
    # A bare "qmx" key left behind by an old hand-typed lesson must not keep attracting more.
    assert normalize_scope("qmx", ["qmx", "me/qmx"], strict=True) == "me/qmx"


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _doc_repo(s: Store, learning_id: int) -> str:
    doc_id = s.get_learning(learning_id).doc_id
    return s._conn.execute("SELECT repo FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]


def test_add_learning_stores_the_canonical_scope(store):
    s, embedder = store
    add_learning(s, embedder, type="howto", statement="seed", scope="me/qmx")
    short = add_learning(s, embedder, type="howto", statement="short name", scope="qmx")
    glob = add_learning(s, embedder, type="howto", statement="typed global", scope="global")
    assert s.get_learning(short).scope == "me/qmx" and _doc_repo(s, short) == "me/qmx"
    assert s.get_learning(glob).scope is None and _doc_repo(s, glob) == "_global"


def test_strict_add_learning_rejects_before_inserting(store):
    s, embedder = store
    with pytest.raises(UnknownScopeError):
        add_learning(s, embedder, type="howto", statement="x", scope="nowhere", strict_scope=True)
    assert s.list_learnings(live_only=False) == []


def test_update_learning_normalizes_a_new_scope(store):
    s, embedder = store
    add_learning(s, embedder, type="howto", statement="seed", scope="me/qmx")
    lid = add_learning(s, embedder, type="howto", statement="was mis-scoped", scope="me/qmx")
    assert update_learning(s, embedder, lid, scope="global").scope is None
    assert _doc_repo(s, lid) == "_global"
    assert update_learning(s, embedder, lid, scope="QMX").scope == "me/qmx"
    with pytest.raises(UnknownScopeError):
        update_learning(s, embedder, lid, scope="nowhere", strict_scope=True)
    assert s.get_learning(lid).scope == "me/qmx"  # unchanged by the rejected update


def test_service_used_by_the_mcp_tool_is_strict(tmp_path):
    embedder = FakeEmbedder(dim=64)
    settings = build_index(tmp_path, embedder, {"a.py": "def f():\n    return 1\n"})
    svc = QmxService(settings, embedder)
    svc.add_learning(type="howto", statement="seed", scope="me/qmx")
    assert svc.add_learning(type="howto", statement="short", scope="qmx")["scope"] == "me/qmx"
    assert svc.add_learning(type="howto", statement="g", scope="global")["scope"] is None
    with pytest.raises(UnknownScopeError):
        svc.add_learning(type="howto", statement="typo", scope="qxm")
