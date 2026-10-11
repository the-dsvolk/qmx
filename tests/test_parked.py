"""Parked items — files are truth, the index is a shadow, and nothing needs the backend to park.

The properties that matter operationally:

- an item is a readable markdown file; closing *moves* it to ``archive/`` (never deletes), and
  ``reopen`` moves it back;
- parking/closing/listing work with the embedding backend down — the shadow index catches up
  through :func:`~qmx.parked.sync_index` (what ``qmx refresh`` runs);
- only **open** items are searchable, and a stale item is flagged, never auto-archived.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from qmx.cli import main
from qmx.config import Settings
from qmx.embed import CircuitBreakerEmbedder, EmbedBackendError
from qmx.mcp_server import build_server
from qmx.parked import (
    KIND,
    ParkedNotFound,
    ParkedRepo,
    doc_path,
    read_item,
    sync_index,
)
from qmx.service import QmxService
from qmx.store import Store
from tests.fakes import FakeEmbedder

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class DownEmbedder(FakeEmbedder):
    """The model backend is unreachable."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbedBackendError("connection refused")


@pytest.fixture
def repo(tmp_path):
    return ParkedRepo(tmp_path / "parked")


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "index.db",
        embed_dim=64,
        embed_model="fake",
        ollama_url="http://127.0.0.1:9",
        parked_root=str(tmp_path / "parked"),
    )


@pytest.fixture
def svc(tmp_path):
    return QmxService(_settings(tmp_path), FakeEmbedder(dim=64))


# -- files -----------------------------------------------------------------------------------------


def test_park_writes_a_readable_markdown_file(repo):
    item = repo.park("Offline BM25 fallback for lessons", why="mirror query degrade", scope="o/r")
    assert item.id == 1
    assert item.path.name == "0001-offline-bm25-fallback-for-lessons.md"
    text = item.path.read_text()
    assert text.startswith("---\nid: 1\nstatus: open\nscope: o/r\n")
    assert "# Offline BM25 fallback for lessons\n\nmirror query degrade" in text
    assert read_item(item.path) == item


def test_ids_are_monotonic_across_open_and_archive(repo):
    a = repo.park("first")
    repo.close(a.id)
    b = repo.park("second")
    assert b.id == a.id + 1  # the archived id is not reused


def test_close_moves_to_archive_with_note_and_reopen_moves_back(repo):
    item = repo.park("rename verbs")
    closed = repo.close(item.id, status="dropped", note="not worth the churn", now=T0)
    assert not item.path.exists()
    assert closed.path.parent == repo.archive
    assert (closed.status, closed.note) == ("dropped", "not worth the churn")
    assert closed.closed == "2026-10-01T12:00:00Z"
    assert repo.items() == []
    assert [i.id for i in repo.items(include_archived=True)] == [item.id]

    reopened = repo.reopen(item.id)
    assert reopened.path.parent == repo.root and reopened.is_open
    assert reopened.closed is None
    assert reopened.note == "not worth the churn"  # history kept
    assert not closed.path.exists()


def test_close_rejects_bad_status_and_unknown_id(repo):
    item = repo.park("x idea")
    with pytest.raises(ValueError):
        repo.close(item.id, status="open")
    with pytest.raises(ParkedNotFound):
        repo.close(999)


def test_empty_idea_is_rejected(repo):
    with pytest.raises(ValueError):
        repo.park("   ")


def test_hand_edited_file_is_tolerated(repo):
    item = repo.park("original title")
    item.path.write_text(
        "---\nid: 1\nstatus: open\nextra: ignored\ncreated: 2026-10-01T12:00:00Z\n---\n"
        "# Edited by hand\n\nmore context here\n"
    )
    got = repo.get(1)
    assert (got.title, got.why, got.scope, got.updated) == (
        "Edited by hand",
        "more context here",
        None,
        "2026-10-01T12:00:00Z",
    )


def test_stale_is_flagged_and_keep_resets_it(repo):
    item = repo.park("old idea", now=T0)
    later = T0 + timedelta(days=31)
    assert item.is_stale(30, now=later)
    assert not item.is_stale(30, now=T0 + timedelta(days=29))
    kept = repo.keep(item.id, now=later)
    assert not kept.is_stale(30, now=later)
    assert kept.created == item.created
    assert repo.get(item.id).is_open  # stale review never archives on its own


def test_closed_items_are_never_stale(repo):
    item = repo.park("old idea", now=T0)
    closed = repo.close(item.id, now=T0)
    assert not closed.is_stale(30, now=T0 + timedelta(days=365))


# -- shadow index ----------------------------------------------------------------------------------


def test_sync_index_indexes_open_and_drops_closed(repo):
    embedder = FakeEmbedder(dim=64)
    a = repo.park("cache tokenizer output")
    b = repo.park("rotate deploy keys")
    with Store.open(":memory:", 64, "fake") as store:
        stats = sync_index(store, embedder, repo)
        assert (stats.indexed, stats.removed) == (2, 0)
        assert sync_index(store, embedder, repo).unchanged == 2  # idempotent

        repo.close(b.id)
        stats = sync_index(store, embedder, repo)
        assert stats.removed == 1
        assert [p for _, p in store.list_documents(KIND)] == [doc_path(a.id)]


# -- service (MCP backing) -------------------------------------------------------------------------


def test_service_park_indexes_and_lists_by_scope(svc):
    svc.park("cache tokenizer output", scope="o/qmx")
    svc.park("global idea about notes")
    svc.park("other repo thing", scope="o/other")

    titles = {i["title"] for i in svc.list_parked(scope="o/qmx")}
    assert titles == {"cache tokenizer output", "global idea about notes"}
    assert len(svc.list_parked()) == 3  # no scope given -> every scope


def test_service_park_reports_similar_open_items(svc):
    first = svc.park("retry embedding requests with backoff", scope="o/r")
    second = svc.park("retry embedding requests with backoff", why="same again", scope="o/r")
    assert second["indexed"] is True
    assert [s["id"] for s in second.get("similar", [])] == [first["id"]]


def test_service_query_ranks_open_items_by_meaning(svc):
    svc.park("rotate deploy keys on the spark")
    target = svc.park("tokenizer cache for faster chunking")
    hits = svc.list_parked(query="tokenizer cache")
    assert hits[0]["id"] == target["id"]


def test_service_unpark_hides_from_search_and_archived_listing_finds_it(svc):
    item = svc.park("tokenizer cache for faster chunking")
    closed = svc.unpark(item["id"], note="landed in #30")
    assert closed["status"] == "done" and closed["note"] == "landed in #30"
    assert svc.list_parked(query="tokenizer cache") == []
    done = svc.list_parked(status="done", query="tokenizer")
    assert [i["id"] for i in done] == [item["id"]]  # keyword fallback over the archive

    reopened = svc.reopen_parked(item["id"])
    assert reopened["indexed"] is True
    assert svc.list_parked(query="tokenizer cache")[0]["id"] == item["id"]


def test_service_list_rejects_unknown_status(svc):
    with pytest.raises(ValueError):
        svc.list_parked(status="pending")


def test_park_works_with_backend_down_and_refresh_catches_up(tmp_path):
    settings = _settings(tmp_path)
    down = QmxService(settings, DownEmbedder(dim=64))
    item = down.park("idea while the spark is off")
    assert item["indexed"] is False
    assert list((tmp_path / "parked").glob("0001-*.md"))
    assert [i["id"] for i in down.list_parked()] == [item["id"]]  # listing is file-only
    closed = down.unpark(item["id"])
    assert closed["status"] == "done"

    second = down.park("another offline idea")
    up = QmxService(settings, FakeEmbedder(dim=64))
    assert up.sync_parked()["indexed"] == 1
    assert up.list_parked(query="offline idea")[0]["id"] == second["id"]


def test_resume_returns_item_and_chat_context(svc):
    item = svc.park("tokenizer cache")
    out = svc.resume_parked(item["id"])
    assert out["item"]["id"] == item["id"]
    assert isinstance(out["context"], list)


# -- MCP + CLI -------------------------------------------------------------------------------------


def test_mcp_registers_parked_tools_and_they_execute(tmp_path):
    settings = _settings(tmp_path)
    server = build_server(settings, QmxService(settings, FakeEmbedder(dim=64)))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"park", "list_parked", "unpark", "resume_parked"} <= names

    def call(name: str, args: dict):
        result = asyncio.run(server.call_tool(name, args))
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    parked = call("park", {"text": "mcp parked idea", "scope": "o/r"})
    assert parked["id"] == 1 and parked["scope"] == "o/r"
    assert call("unpark", {"item_id": 1, "status": "dropped"})["status"] == "dropped"


def test_cli_park_list_unpark_round_trip(tmp_path, monkeypatch, capsys):
    settings = _settings(tmp_path)
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, *a, **k: settings))
    fake = CircuitBreakerEmbedder(FakeEmbedder(dim=64))
    monkeypatch.setattr("qmx.service.read_embedder", lambda s: fake)

    assert main(["park", "cli", "parked", "idea", "--why", "ctx", "--scope", "o/r"]) == 0
    assert "parked #1 (o/r): cli parked idea" in capsys.readouterr().out

    assert main(["parked", "--scope", "o/r"]) == 0
    out = capsys.readouterr().out
    assert "#1" in out and "cli parked idea" in out and "ctx" in out

    assert main(["unpark", "1", "--note", "done it"]) == 0
    assert "unparked #1 [done]" in capsys.readouterr().out
    assert main(["parked", "--scope", "o/r"]) == 0
    assert "nothing parked" in capsys.readouterr().out
    assert main(["unpark", "42"]) == 1
