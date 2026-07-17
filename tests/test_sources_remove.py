"""Project management: list_sources and remove_source (file + subtree)."""

from __future__ import annotations

import pytest

from qmx.index import index_paths
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder(dim=64)
    proj = (tmp_path / "proj").resolve()
    (proj / "sub").mkdir(parents=True)
    (proj / "a.py").write_text("def alpha():\n    return 1\n")
    (proj / "sub" / "b.py").write_text("def beta():\n    return 2\n")
    (proj / "sub" / "c.py").write_text("def gamma():\n    return 3\n")
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    index_paths([proj], store, embedder)
    yield proj, store, embedder
    store.close()


def test_list_sources(env):
    proj, store, _ = env
    sources = store.list_sources()
    assert len(sources) == 1
    src = sources[0]
    assert src["repo"] == "proj"
    assert src["documents"] == 3
    assert src["chunks"] == 3
    assert src["sample_path"].endswith(".py")


def test_remove_single_file(env):
    proj, store, embedder = env
    docs, orphaned = store.remove_source(str(proj / "a.py"))
    assert docs == 1
    assert orphaned == 1
    assert store.counts()["documents"] == 2
    hits = search(store, embedder, "alpha", k=5)
    assert all("a.py" not in (h.hit.path or "") for h in hits)


def test_remove_subtree(env):
    proj, store, _ = env
    docs, orphaned = store.remove_source(str(proj / "sub"))
    assert docs == 2  # b.py + c.py
    assert orphaned == 2
    assert store.counts()["documents"] == 1  # only a.py left
    assert store.index_stats()["tombstoned_chunks"] == 2
    assert store.purge_orphans() == 2


def test_remove_whole_project(env):
    proj, store, _ = env
    docs, _ = store.remove_source(str(proj))
    assert docs == 3
    assert store.counts()["documents"] == 0


def test_remove_nonexistent(env):
    _, store, _ = env
    assert store.remove_source("/nope/not/indexed") == (0, 0)
