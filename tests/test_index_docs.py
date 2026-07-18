"""Repo indexing of markdown docs (kind='doc') alongside code, via chunk_markdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from qmx.index import index_paths, repo_kind
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder

CODE = "def alpha(x):\n    return x + 1\n"
MD = "# CAPI Glossary\n\nEgress is outbound bandwidth.\n\n## Quota\n\nA per-bucket cap.\n"


def test_repo_kind_routing():
    assert repo_kind(Path("a/b.py")) == "code"
    assert repo_kind(Path("kb/GLOSSARY.md")) == "doc"
    assert repo_kind(Path("readme.markdown")) == "doc"
    assert repo_kind(Path("data.csv")) is None
    assert repo_kind(Path("notes.txt")) is None


@pytest.fixture
def repo(tmp_path):
    embedder = FakeEmbedder(dim=64)
    (tmp_path / "app.py").write_text(CODE)
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "GLOSSARY.md").write_text(MD)
    (tmp_path / "data.csv").write_text("x,y\n1,2\n")  # ignored (not code/doc)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    yield tmp_path, store, embedder
    store.close()


def test_markdown_indexed_as_doc(repo):
    tmp_path, store, embedder = repo
    stats = index_paths([tmp_path], store, embedder)
    assert stats.files_indexed == 2  # app.py + GLOSSARY.md (not data.csv)

    import sqlite3

    c = sqlite3.connect(tmp_path / "index.db")
    kinds = dict(c.execute("SELECT kind, count(*) FROM documents GROUP BY kind").fetchall())
    assert kinds == {"code": 1, "doc": 1}

    # the kb markdown is now searchable (kind=doc)
    [qvec] = embedder.embed(["egress bandwidth quota"])
    hits = store.search_vec(qvec, k=5, kind="doc")
    assert hits and all(h.kind == "doc" for h in hits)
    assert hits[0].path.endswith("GLOSSARY.md")


def test_doc_returned_by_unified_query(repo):
    tmp_path, store, embedder = repo
    index_paths([tmp_path], store, embedder)
    results = search(store, embedder, "outbound bandwidth egress", k=5)  # no kind filter
    assert any((r.hit.path or "").endswith("GLOSSARY.md") for r in results)


def test_deleted_markdown_is_pruned(repo):
    tmp_path, store, embedder = repo
    index_paths([tmp_path], store, embedder)
    (tmp_path / "kb" / "GLOSSARY.md").unlink()
    stats = index_paths([tmp_path], store, embedder)
    assert stats.files_removed == 1  # the doc was pruned

    import sqlite3

    c = sqlite3.connect(tmp_path / "index.db")
    docs = c.execute("SELECT count(*) FROM documents WHERE kind='doc'").fetchone()[0]
    assert docs == 0
