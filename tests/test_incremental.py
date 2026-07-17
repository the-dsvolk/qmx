"""Phase 2 acceptance: edit re-embeds only changed chunks; delete removes them; dedup + reuse."""

from __future__ import annotations

import pytest

from qmx.index import index_paths
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder

TWO_FUNCS = """\
def alpha(x):
    return x + 1


def beta(y):
    return y * 2
"""

TWO_FUNCS_BETA_EDITED = """\
def alpha(x):
    return x + 1


def beta(y):
    return y * 3
"""

OTHER = """\
def gamma(z):
    return z - 1
"""

SHARED = """\
def shared():
    return 42
"""


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    yield tmp_path, store, embedder
    store.close()


def test_edit_reembeds_only_changed_chunk(env):
    root, store, embedder = env
    (root / "a.py").write_text(TWO_FUNCS)
    (root / "b.py").write_text(OTHER)
    first = index_paths([root], store, embedder)
    assert first.chunks_embedded == 3  # alpha, beta, gamma — all new

    # Edit only beta's body.
    (root / "a.py").write_text(TWO_FUNCS_BETA_EDITED)
    second = index_paths([root], store, embedder)
    assert second.files_indexed == 1  # only a.py
    assert second.files_skipped == 1  # b.py unchanged (file_hash)
    assert second.chunks_embedded == 1  # only the edited beta
    assert second.chunks_reused >= 1  # alpha unchanged
    assert second.chunks_orphaned == 1  # old beta tombstoned


def test_unchanged_run_embeds_nothing(env):
    root, store, embedder = env
    (root / "a.py").write_text(TWO_FUNCS)
    index_paths([root], store, embedder)
    again = index_paths([root], store, embedder)
    assert again.files_indexed == 0
    assert again.files_skipped == 1
    assert again.chunks_embedded == 0


def test_delete_file_removes_its_chunks_from_search(env):
    root, store, embedder = env
    (root / "a.py").write_text(TWO_FUNCS)
    (root / "b.py").write_text(OTHER)
    index_paths([root], store, embedder)
    live_before = store.index_stats()["live_chunks"]

    (root / "a.py").unlink()
    stats = index_paths([root], store, embedder)
    assert stats.files_removed == 1
    assert stats.chunks_orphaned == 2  # alpha + beta

    # gone from search...
    [qvec] = embedder.embed(["alpha return"])
    assert all("a.py" not in (h.hit.path or "") for h in search(store, embedder, "alpha", k=10))
    assert store.index_stats()["live_chunks"] == live_before - 2

    # ...still on disk as warm tombstones until gc
    assert store.index_stats()["tombstoned_chunks"] == 2
    assert store.purge_orphans() == 2
    assert store.index_stats()["tombstoned_chunks"] == 0


def test_identical_code_across_files_dedups(env):
    root, store, embedder = env
    (root / "c.py").write_text(SHARED)
    (root / "d.py").write_text(SHARED)
    stats = index_paths([root], store, embedder)
    assert stats.chunks_embedded == 1  # embedded once despite two files
    assert store.counts()["chunks"] == 1
    assert store.index_stats()["mentions"] == 2

    # deleting one file keeps the shared chunk alive (still mentioned by the other)
    (root / "c.py").unlink()
    index_paths([root], store, embedder)
    assert store.index_stats()["live_chunks"] == 1


def test_rename_reuses_warm_embedding(env):
    root, store, embedder = env
    (root / "e.py").write_text(SHARED)
    index_paths([root], store, embedder)

    (root / "e.py").unlink()
    index_paths([root], store, embedder)  # orphaned, NOT purged
    assert store.index_stats()["tombstoned_chunks"] == 1

    (root / "f.py").write_text(SHARED)  # same content, new file (a rename)
    stats = index_paths([root], store, embedder)
    assert stats.chunks_embedded == 0  # warm tombstone reused, no re-embed
    assert stats.chunks_reused == 1
    assert store.index_stats()["live_chunks"] == 1
