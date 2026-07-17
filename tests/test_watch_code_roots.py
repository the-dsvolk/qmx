"""`qmx watch` defaults to code_roots, and the store sets a busy_timeout for concurrent writers."""

from __future__ import annotations

from pathlib import Path

from qmx.cli import _watch_targets
from qmx.config import Settings
from qmx.store import Store
from tests.fakes import FakeEmbedder


def _settings(code_roots: tuple[str, ...]) -> Settings:
    return Settings(embed_dim=64, embed_model="fake", code_roots=code_roots)


def test_watch_uses_explicit_paths_when_given():
    s = _settings(("~/GitHub/Cruise/xtorch",))
    assert _watch_targets(s, ["/a", "/b"]) == [Path("/a"), Path("/b")]


def test_watch_falls_back_to_code_roots():
    s = _settings(("~/GitHub/Cruise/xtorch", "~/GitHub/Cruise/cpe-intelligence"))
    got = _watch_targets(s, [])
    assert got == [
        Path.home() / "GitHub/Cruise/xtorch",
        Path.home() / "GitHub/Cruise/cpe-intelligence",
    ]  # ~ expanded


def test_watch_empty_when_no_paths_and_no_code_roots():
    assert _watch_targets(_settings(()), []) == []


def test_store_sets_busy_timeout(tmp_path):
    embedder = FakeEmbedder(dim=64)
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as store:
        (timeout,) = store._conn.execute("PRAGMA busy_timeout").fetchone()
        assert timeout == 5000
