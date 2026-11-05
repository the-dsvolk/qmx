"""``qmx reembed``: rebuild only the vector table, from stored chunk text, resumably."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from qmx import cli
from qmx.config import Settings
from qmx.index import index_paths
from qmx.learnings import add_learning
from qmx.reembed import reembed
from qmx.search import search
from qmx.store import Store, StoreSchemaMismatch, load_sqlite_vec
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.embedded = 0

    def embed(self, texts):
        self.embedded += len(texts)
        return super().embed(texts)


def _build(tmp_path: Path) -> tuple[Path, int]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "net.py").write_text(
        "def retry_with_backoff(func):\n    return func()\n\n\n"
        "def parse_header(line):\n    return line.split(':')\n"
    )
    (repo / "notes.md").write_text("# Wash sales\n\nA buy within 30 days disallows the loss.\n")
    db = tmp_path / "index.db"
    old = FakeEmbedder(dim=64)
    with Store.open(db, old.dim, "old-model") as store:
        index_paths([repo], store, old)
        add_learning(store, old, type="howto", scope=None,
                     statement="Check the trading window before any sale.")
        chunks = store.counts()["chunks"]
    return db, chunks


def _meta(db: Path) -> dict[str, str]:
    con = sqlite3.connect(db)
    try:
        return dict(con.execute("SELECT key, value FROM meta").fetchall())
    finally:
        con.close()


def test_reembed_rebuilds_vectors_and_meta_only(tmp_path):
    db, chunks = _build(tmp_path)
    new = CountingEmbedder(dim=32)
    stats = reembed(db, new, "new-model")
    assert stats.rebuilt_table and (stats.old_model, stats.old_dim) == ("old-model", 64)
    assert stats.embedded == new.embedded == stats.chunks_total == chunks
    assert _meta(db)["embed_model"] == "new-model" and _meta(db)["embed_dim"] == "32"

    with pytest.raises(StoreSchemaMismatch):  # the old config can no longer open it
        Store.open(db, 64, "old-model")
    with Store.open(db, 32, "new-model") as store:
        counts = store.counts()
        assert counts["vectors"] == counts["chunks"] == chunks
        hits = search(store, new, "retry backoff", k=3)
        assert hits and hits[0].hit.path.endswith("net.py")
        assert store.list_learnings()  # lessons and their docs came across untouched


def test_reembed_resumes_without_redoing_finished_chunks(tmp_path):
    db, chunks = _build(tmp_path)
    reembed(db, FakeEmbedder(dim=32), "new-model")
    con = sqlite3.connect(db)
    load_sqlite_vec(con)
    with con:
        con.execute("DELETE FROM vec_chunks WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT 2)")
    con.close()

    again = CountingEmbedder(dim=32)
    stats = reembed(db, again, "new-model")
    assert not stats.rebuilt_table
    assert again.embedded == stats.embedded == 2
    assert stats.already_embedded == chunks - 2


def test_cli_reembed_refuses_the_live_index_without_in_place(tmp_path, monkeypatch, capsys):
    db, _ = _build(tmp_path)
    settings = Settings(db_path=db, embed_dim=32, embed_model="new-model")
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, *a, **k: settings))
    assert cli.main(["reembed", "--db", str(db)]) == 2
    assert "refusing to re-embed the configured index" in capsys.readouterr().err
    assert _meta(db)["embed_model"] == "old-model"  # untouched


def test_cli_reembed_converts_a_copy(tmp_path, monkeypatch, capsys):
    db, chunks = _build(tmp_path)
    copy = tmp_path / "copy.db"
    src = sqlite3.connect(db)
    dst = sqlite3.connect(copy)
    src.backup(dst)
    src.close()
    dst.close()
    settings = Settings(db_path=db, embed_dim=32, embed_model="new-model")
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, *a, **k: settings))
    monkeypatch.setattr(
        cli, "make_embedder", lambda _s: contextlib.nullcontext(FakeEmbedder(dim=32))
    )
    assert cli.main(["reembed", "--db", str(copy)]) == 0
    assert f"re-embedded {chunks} of {chunks} chunk(s)" in capsys.readouterr().out
    assert _meta(copy)["embed_model"] == "new-model"
    assert _meta(db)["embed_model"] == "old-model"


def test_reembed_reuses_vectors_from_an_earlier_conversion(tmp_path):
    db, chunks = _build(tmp_path)
    first = tmp_path / "first.db"
    _backup(db, first)
    reembed(first, FakeEmbedder(dim=32), "new-model")

    # The live index moves on: one more file indexed after the first conversion.
    (tmp_path / "repo" / "extra.py").write_text("def brand_new_helper():\n    return 42\n")
    with Store.open(db, 64, "old-model") as store:
        index_paths([tmp_path / "repo"], store, FakeEmbedder(dim=64))
        grown = store.counts()["chunks"]
    assert grown > chunks

    final = tmp_path / "final.db"
    _backup(db, final)
    again = CountingEmbedder(dim=32)
    stats = reembed(final, again, "new-model", reuse_from=first)
    assert stats.reused == chunks
    assert again.embedded == stats.embedded == grown - chunks
    with Store.open(final, 32, "new-model") as store:
        assert store.counts()["vectors"] == grown
        hits = search(store, again, "brand new helper", k=3)
        assert hits and hits[0].hit.path.endswith("extra.py")


def test_reembed_rejects_a_reuse_index_of_another_model(tmp_path):
    db, _ = _build(tmp_path)
    copy = tmp_path / "copy.db"
    _backup(db, copy)
    with pytest.raises(ValueError, match="holds 'old-model'"):
        reembed(copy, FakeEmbedder(dim=32), "new-model", reuse_from=db)


def _backup(src_path: Path, dst_path: Path) -> None:
    src, dst = sqlite3.connect(src_path), sqlite3.connect(dst_path)
    src.backup(dst)
    src.close()
    dst.close()
