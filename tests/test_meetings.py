"""Memoir call folders indexed as ``kind='meeting'`` — chunking, indexing, search, MCP tool."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from qmx.chunk.meeting import (
    MeetingInfo,
    chunk_meeting_markdown,
    chunk_transcript,
    parse_attendees,
    read_meeting_info,
)
from qmx.config import Settings
from qmx.index import backfill_meetings, index_meeting, is_meeting_folder
from qmx.mcp_server import build_server
from qmx.service import QmxService
from qmx.store import Store
from tests.fakes import FakeEmbedder

HEADER = "Weekly sync · 2026-10-18 14:30 · Alice, Bob"


def _transcript(n: int, step: int = 20) -> str:
    """``n`` segment lines, one every ``step`` seconds, alternating speakers."""
    lines = []
    for i in range(n):
        t = i * step
        who = "Me" if i % 2 == 0 else "Them"
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {who}: point {i} about the vendor budget")
    return "\n".join(lines) + "\n"


def _write_call(root, name="2026-10-18_weekly-sync", *, ended=True, attendees="Alice, Bob"):
    folder = root / name
    folder.mkdir(parents=True)
    meta = {"name": "Weekly sync", "started_at": "2026-10-18T14:30:05-07:00"}
    if ended:
        meta["ended_at"] = "2026-10-18T15:00:00-07:00"
    (folder / "meeting.json").write_text(json.dumps(meta))
    (folder / "transcript.txt").write_text(_transcript(12))
    (folder / "summary.md").write_text(
        "# Weekly sync — summary\n\n## Decisions\n\n- Dedicated interconnect replaces Flexify.\n"
    )
    (folder / "notes.md").write_text("# Notes — Weekly sync\n\n- [01:10] check egress quota\n")
    prep = "# Prep — Weekly sync\n\n## Before the call\n\n"
    prep += f"Attendees: {attendees}\n" if attendees else "_(none)_\n"
    (folder / "prep.md").write_text(prep)
    return folder


# -- chunking ------------------------------------------------------------------------------------


def test_transcript_windows_are_60_to_90_seconds_with_one_line_overlap():
    chunks = chunk_transcript(_transcript(12), HEADER)  # 0:00 .. 3:40, one line every 20 s
    assert len(chunks) >= 3
    for c in chunks:
        assert c.text.startswith(HEADER + "\n")
    # Each window spans 60–90 s and the next one starts on the previous window's last line.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_line == prev.end_line
    first = chunks[0]
    assert (first.start_line, first.end_line) == (1, 4)  # 00:00 .. 01:00
    assert first.symbol == "Me/Them 00:00–01:00"
    assert chunks[-1].end_line == 12  # nothing dropped at the end


def test_transcript_window_breaks_on_size_between_lines():
    long_line = "x" * 900
    text = f"[00:00] Me: {long_line}\n[00:05] Them: {long_line}\n[00:10] Me: short\n"
    chunks = chunk_transcript(text)
    assert all(c.start_line == c.end_line or "\n" in c.text for c in chunks)
    assert [c.start_line for c in chunks][:2] == [1, 2]  # never split mid-line


def test_transcript_hour_clock_and_unparsed_lines():
    text = "[59:50] Me: almost an hour\n[1:00:10] Them: past the hour\ncontinued line\n"
    (chunk,) = chunk_transcript(text)
    assert chunk.symbol == "Me/Them 59:50–1:00:10"
    assert chunk.end_line == 3  # the stray line rides with the segment before it


def test_markdown_chunks_get_the_header():
    chunks = chunk_meeting_markdown("# Summary\n\nWe chose L4.\n", HEADER)
    assert chunks[0].text.startswith(HEADER + "\n# Summary")
    assert chunks[0].symbol == "Summary"


def test_parse_attendees_and_meeting_info(tmp_path):
    assert parse_attendees("## Before the call\n- Attendees: Alice (PM); Bob\n") == (
        "Alice (PM)",
        "Bob",
    )
    assert parse_attendees("_(none)_\n") == ()

    folder = _write_call(tmp_path)
    info = read_meeting_info(folder)
    assert info == MeetingInfo("Weekly sync", "2026-10-18", "14:30", ("Alice", "Bob"))
    assert info.header() == HEADER


def test_meeting_info_falls_back_to_folder_name(tmp_path):
    folder = tmp_path / "2026-10-19_team-meeting"
    folder.mkdir()
    (folder / "meeting.json").write_text("{}")
    assert read_meeting_info(folder).header() == "team meeting · 2026-10-19"


# -- indexing ------------------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    yield tmp_path, store, embedder
    store.close()


def _docs(db):
    c = sqlite3.connect(db)
    try:
        return c.execute("SELECT kind, repo, path FROM documents ORDER BY path").fetchall()
    finally:
        c.close()


def test_index_meeting_writes_one_meeting_doc_per_file(env):
    tmp_path, store, embedder = env
    folder = _write_call(tmp_path / "memoir")
    stats = index_meeting(folder, store, embedder)
    assert stats.files_indexed == 4

    docs = _docs(tmp_path / "index.db")
    assert {d[0] for d in docs} == {"meeting"}
    assert {d[1] for d in docs} == {"2026-10-18_weekly-sync"}
    assert sorted(p.rsplit("/", 1)[1] for _, _, p in docs) == [
        "notes.md",
        "prep.md",
        "summary.md",
        "transcript.txt",
    ]

    again = index_meeting(folder, store, embedder)
    assert again.files_indexed == 0 and again.files_skipped == 4


def test_new_attendees_reindex_every_file(env):
    tmp_path, store, embedder = env
    folder = _write_call(tmp_path / "memoir")
    index_meeting(folder, store, embedder)
    (folder / "prep.md").write_text("## Before the call\n\nAttendees: Alice, Bob, Carol\n")
    stats = index_meeting(folder, store, embedder)
    assert stats.files_indexed == 4  # the header changed, so every file's chunks did


def test_removed_file_is_dropped(env):
    tmp_path, store, embedder = env
    folder = _write_call(tmp_path / "memoir")
    index_meeting(folder, store, embedder)
    (folder / "summary.md").unlink()
    stats = index_meeting(folder, store, embedder)
    assert stats.files_removed == 1
    assert not any(p.endswith("summary.md") for _, _, p in _docs(tmp_path / "index.db"))


def test_call_in_progress_is_skipped(env):
    tmp_path, store, embedder = env
    folder = _write_call(tmp_path / "memoir", ended=False)
    stats = index_meeting(folder, store, embedder)
    assert stats.files_indexed == 0 and stats.files_skipped == 1
    assert _docs(tmp_path / "index.db") == []


def test_backfill_indexes_only_call_folders(env):
    tmp_path, store, embedder = env
    save = tmp_path / "memoir"
    _write_call(save, "2026-10-18_weekly-sync")
    _write_call(save, "2026-10-19_team-meeting")
    (save / "not-a-call").mkdir()
    assert not is_meeting_folder(save / "not-a-call")

    stats = backfill_meetings(save, store, embedder)
    assert stats.files_indexed == 8
    assert {d[1] for d in _docs(tmp_path / "index.db")} == {
        "2026-10-18_weekly-sync",
        "2026-10-19_team-meeting",
    }


# -- search + MCP --------------------------------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    embedder = FakeEmbedder(dim=64)
    db = tmp_path / "index.db"
    (tmp_path / "app.py").write_text("def flexify_client():\n    return 'interconnect'\n")
    with Store.open(db, embedder.dim, "fake") as store:
        index_meeting(_write_call(tmp_path / "memoir"), store, embedder)
        from qmx.index import index_paths

        index_paths([tmp_path / "app.py"], store, embedder)
    settings = Settings(
        db_path=db, embed_dim=embedder.dim, embed_model="fake", ollama_url="http://127.0.0.1:9"
    )
    return settings, QmxService(settings, embedder)


def test_meetings_search_names_the_call(service):
    _, svc = service
    hits = svc.meetings("Flexify dedicated interconnect decision", k=3)
    assert hits, "expected a meeting hit"
    assert {h["kind"] for h in hits} == {"meeting"}  # code mentioning flexify is excluded
    top = hits[0]
    assert top["meeting"] == "2026-10-18_weekly-sync"
    assert top["date"] == "2026-10-18"
    assert top["file"] == "summary.md"
    assert top["text"].startswith(HEADER)


def test_meetings_mcp_tool_is_registered(service):
    settings, svc = service
    server = build_server(settings, svc)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "meetings" in names
    result = asyncio.run(server.call_tool("meetings", {"text": "vendor budget", "k": 2}))
    assert result is not None
