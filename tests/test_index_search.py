"""Phase 1 acceptance: index a repo; a known function is in top-5 for a by-meaning query."""

from __future__ import annotations

import pytest

from qmx.embed import EmbedBackendError
from qmx.index import IndexStats, index_paths, iter_source_files
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder

NET_PY = '''\
import time


def retry_with_backoff(func, attempts=5):
    """Retry a callable with exponential backoff between failed attempts."""
    delay = 0.5
    for attempt in range(attempts):
        try:
            return func()
        except Exception:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("all retry attempts failed")
'''

MATH_PY = """\
def add(a, b):
    return a + b


def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""

STRINGS_PY = """\
def slugify(text):
    return "-".join(text.lower().split())
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "net.py").write_text(NET_PY)
    (tmp_path / "math_utils.py").write_text(MATH_PY)
    (tmp_path / "strings.py").write_text(STRINGS_PY)
    # junk that must be pruned / ignored
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "README.md").write_text("# not code\n")
    return tmp_path


@pytest.fixture
def indexed(repo):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(repo / "index.db", embedder.dim, "fake")
    stats = index_paths([repo], store, embedder)
    yield store, embedder, stats
    store.close()


def test_iter_source_files_prunes_junk_indexes_code_and_md(repo):
    files = {p.name for p in iter_source_files(repo)}
    # code + markdown are indexed; .git (dotdir) is pruned, data/binaries ignored
    assert files == {"net.py", "math_utils.py", "strings.py", "README.md"}


def test_index_stats(indexed):
    _, _, stats = indexed
    assert isinstance(stats, IndexStats)
    assert stats.files_indexed == 4  # 3 .py + README.md
    assert stats.chunks_added >= 4  # 4 functions across the files
    assert stats.errors == []


def test_known_function_in_top5_for_meaning_query(indexed):
    store, embedder, _ = indexed
    results = search(store, embedder, "how to retry a failed network request", k=5)
    symbols = [r.hit.symbol for r in results]
    assert "retry_with_backoff" in symbols
    assert results[0].hit.symbol == "retry_with_backoff"  # and it ranks first
    assert results[0].hit.path.endswith("net.py")


def test_reindex_unchanged_is_skipped(indexed, repo):
    store, embedder, _ = indexed
    stats2 = index_paths([repo], store, embedder)
    assert stats2.files_indexed == 0
    assert stats2.files_skipped == 4
    assert stats2.chunks_added == 0


def test_reindex_changed_file_replaces_chunks(indexed, repo):
    store, embedder, _ = indexed
    before = store.counts()["chunks"]
    (repo / "strings.py").write_text(
        'def slugify(text):\n    return "-".join(text.lower().split())\n\n\n'
        "def shout(text):\n    return text.upper()\n"
    )
    stats2 = index_paths([repo], store, embedder)
    assert stats2.files_indexed == 1
    assert store.counts()["chunks"] == before + 1  # one new function added


def test_failed_embed_leaves_file_reprocessable(repo):
    class BoomEmbedder:
        dim = 64

        def embed(self, texts):
            raise EmbedBackendError("backend down")

    store = Store.open(repo / "index.db", 64, "fake")
    with pytest.raises(EmbedBackendError):
        index_paths([repo], store, BoomEmbedder())
    # No document row was persisted, so nothing is marked "indexed" -> a later run reprocesses it.
    assert store.counts()["documents"] == 0
    assert store.document_hash("code", str((repo / "net.py").resolve())) is None
    store.close()
