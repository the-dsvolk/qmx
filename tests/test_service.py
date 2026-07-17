"""QmxService — the read API behind the MCP tools."""

from __future__ import annotations

import pytest

from qmx.service import QmxService
from tests.fakes import FakeEmbedder, build_index

FILES = {
    "net.py": (
        "import time\n\n\n"
        "def retry_with_backoff(func, attempts=5):\n"
        '    """Retry a callable with exponential backoff between failed attempts."""\n'
        "    return func()\n"
    ),
    "math_utils.py": "def add(a, b):\n    return a + b\n",
}


@pytest.fixture
def service(tmp_path):
    embedder = FakeEmbedder(dim=64)
    settings = build_index(tmp_path, embedder, FILES)
    return QmxService(settings, embedder)


def test_query_returns_ranked_hits(service):
    hits = service.query("retry a failed request with backoff", k=5)
    assert hits
    top = hits[0]
    assert top["symbol"] == "retry_with_backoff"
    assert top["path"].endswith("net.py")
    assert "score" in top and top["start_line"] is not None


def test_query_kind_filter(service):
    assert service.query("anything", k=5, kind="chat") == []  # no chat docs indexed


def test_get_returns_full_chunk(service):
    hits = service.query("retry backoff", k=3)
    got = service.get(hits[0]["chunk_id"])
    assert got is not None
    assert got["chunk_id"] == hits[0]["chunk_id"]
    assert "def retry_with_backoff" in got["text"]


def test_get_missing_returns_none(service):
    assert service.get(999_999) is None


def test_status_reports_stats_and_backend(service):
    st = service.status()
    assert st["index"]["live_chunks"] >= 2
    assert st["embed_model"] == "fake"
    assert st["ollama_ok"] is False  # dead URL in the test settings
