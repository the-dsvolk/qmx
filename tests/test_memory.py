"""Memory indexing: markdown chunker, kind='memory' ingest via globs, and config."""

from __future__ import annotations

import pytest

from qmx.chunk.doc import chunk_markdown
from qmx.config import Settings
from qmx.index import index_memory, index_memory_dir
from qmx.store import Store
from tests.fakes import FakeEmbedder

MD = """\
# DGX Spark runtime

Ollama runs rootless on the Spark.

## SSH connect

Connect with `ssh spark-0e81.local` as user dsvolk.

## GPU note

The GB10 needs the cuda_v13 runner.
"""


def test_chunk_markdown_splits_on_headings():
    chunks = chunk_markdown(MD)
    symbols = [c.symbol for c in chunks]
    assert symbols == ["DGX Spark runtime", "SSH connect", "GPU note"]
    ssh = next(c for c in chunks if c.symbol == "SSH connect")
    assert "ssh spark-0e81.local" in ssh.text
    assert ssh.start_line == 5
    assert all(c.start_line is not None for c in chunks)


def test_chunk_markdown_preamble_before_first_heading():
    chunks = chunk_markdown("intro text with no heading\n\nmore intro\n\n# Later\nbody")
    assert chunks[0].symbol is None
    assert "intro text" in chunks[0].text
    assert chunks[1].symbol == "Later"


def test_chunk_markdown_splits_long_section():
    big = "# Big\n" + ("filler paragraph. " * 200)  # > 1500 chars
    chunks = chunk_markdown(big)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1500 for c in chunks)


def test_chunk_markdown_empty():
    assert chunk_markdown("   \n\n  ") == []


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    yield store, embedder, tmp_path
    store.close()


def test_index_memory_via_globs(env):
    store, embedder, tmp_path = env
    mem = tmp_path / "projects" / "proj-a" / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("# Index\n- spark runtime\n")
    (mem / "spark.md").write_text(MD)
    globs = [str(tmp_path / "projects" / "*" / "memory")]

    stats = index_memory(globs, store, embedder)
    assert stats.files_indexed == 2
    assert stats.chunks_embedded >= 4

    [qvec] = embedder.embed(["how to ssh into the spark"])
    hits = store.search_vec(qvec, k=5, kind="memory")
    assert hits and all(h.kind == "memory" for h in hits)


def test_index_memory_dir_used_by_capture(env):
    store, embedder, tmp_path = env
    mem = tmp_path / "session-dir" / "memory"
    mem.mkdir(parents=True)
    (mem / "note.md").write_text(MD)
    stats = index_memory_dir(mem, store, embedder)
    assert stats.files_indexed == 1
    # a missing memory dir is a no-op (capture on a project with no memory/)
    assert index_memory_dir(tmp_path / "nope", store, embedder).files_indexed == 0


def test_memory_globs_config_default_and_override():
    from pathlib import Path

    d = Settings.load(config_path=Path("/does/not/exist"), env={})
    assert d.memory_globs == ("~/.claude/projects/*/memory",)
    s = Settings.load(
        config_path=Path("/does/not/exist"),
        env={"QMX_MEMORY_GLOBS": "~/.claude/projects/*/memory, ~/.claude/CLAUDE.md"},
    )
    assert s.memory_globs == ("~/.claude/projects/*/memory", "~/.claude/CLAUDE.md")
