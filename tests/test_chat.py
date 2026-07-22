"""Chat ingest: transcript parsing (drop noise), kind='chat' indexing, and recall."""

from __future__ import annotations

import json

import pytest

from qmx.chunk.chat import chunk_chat
from qmx.index import backfill_chats, index_transcript
from qmx.service import QmxService
from qmx.store import Store
from tests.fakes import FakeEmbedder


def _line(**kw) -> str:
    return json.dumps(kw)


TRANSCRIPT = "\n".join(
    [
        _line(type="summary", summary="noise"),  # non-message type -> skipped
        _line(type="user", isSidechain=False,
              message={"role": "user", "content": "how do I retry a failed network request"}),
        _line(type="assistant", isSidechain=False, message={"role": "assistant", "content": [
            {"type": "thinking", "thinking": "internal reasoning that must be dropped"},
            {"type": "text", "text": "Use exponential backoff between retries."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "echo secret"}},
        ]}),
        _line(type="user", message={"role": "user", "content": [
            {"type": "tool_result", "content": "TOOL OUTPUT that must be dropped"},
        ]}),  # only a tool_result -> no text -> dropped
        _line(type="assistant", isSidechain=True, message={"role": "assistant", "content": [
            {"type": "text", "text": "subagent side-channel chatter"},
        ]}),  # side-chain -> skipped
        _line(type="user", message={"role": "user",
              "content": "<system-reminder>ignore me</system-reminder>what about timeouts"}),
    ]
)  # fmt: skip


def test_chunk_chat_keeps_only_clean_turns():
    chunks = chunk_chat(TRANSCRIPT)
    texts = [c.text for c in chunks]
    roles = [c.symbol for c in chunks]
    assert roles == ["user", "assistant", "user"]
    assert texts[0] == "how do I retry a failed network request"
    assert texts[1] == "Use exponential backoff between retries."
    assert texts[2] == "what about timeouts"  # system-reminder stripped
    joined = "\n".join(texts)
    for noise in ["internal reasoning", "echo secret", "TOOL OUTPUT", "side-channel", "ignore me"]:
        assert noise not in joined
    assert all(c.start_line is not None for c in chunks)


# Cursor schema: top-level ``role`` (not ``type``), ``message`` has no inner role, marker lines are
# ``{"type": "turn_ended", ...}``, and no tool_result/thinking blocks are emitted. Mirrors the real
# ~/.cursor/projects/*/agent-transcripts/<uuid>/<uuid>.jsonl format verified during the port.
CURSOR_TRANSCRIPT = "\n".join(
    [
        _line(role="user", message={"content": [
            {"type": "text", "text": "how do I retry a failed network request"},
        ]}),
        _line(role="assistant", message={"content": [
            {"type": "text", "text": "Use exponential backoff between retries."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "echo secret"}},
        ]}),
        _line(type="turn_ended", status="success"),  # marker line (no role) -> skipped
        _line(role="assistant", message={"content": [
            {"type": "tool_use", "name": "Glob", "input": {"glob_pattern": "*.py"}},
        ]}),  # only tool_use -> no text -> dropped
        _line(role="user", message={"content": [
            {"type": "text",
             "text": "<system-reminder>ignore me</system-reminder>what about timeouts"},
        ]}),
    ]
)  # fmt: skip


def test_chunk_chat_cursor_source_keeps_only_clean_turns():
    chunks = chunk_chat(CURSOR_TRANSCRIPT, source="cursor")
    texts = [c.text for c in chunks]
    roles = [c.symbol for c in chunks]
    assert roles == ["user", "assistant", "user"]
    assert texts[0] == "how do I retry a failed network request"
    assert texts[1] == "Use exponential backoff between retries."
    assert texts[2] == "what about timeouts"  # system-reminder stripped
    joined = "\n".join(texts)
    for noise in ["echo secret", "turn_ended", "glob_pattern", "ignore me"]:
        assert noise not in joined
    assert all(c.start_line is not None for c in chunks)


def test_claude_and_cursor_sources_do_not_cross_parse():
    # Claude parser sees no top-level ``type=user/assistant`` in a Cursor transcript -> nothing.
    assert chunk_chat(CURSOR_TRANSCRIPT, source="claude") == []
    # Cursor parser sees no top-level ``role`` in a Claude transcript -> nothing.
    assert chunk_chat(TRANSCRIPT, source="cursor") == []


def test_index_transcript_cursor_source(store):
    s, embedder, tmp_path = store
    tp = tmp_path / "cursor-session.jsonl"
    tp.write_text(CURSOR_TRANSCRIPT)
    stats = index_transcript(tp, s, embedder, source="cursor")
    assert stats.files_indexed == 1
    assert stats.chunks_embedded == 3  # 3 clean turns
    [qvec] = embedder.embed(["retry with backoff"])
    hits = s.search_vec(qvec, k=5, kind="chat")
    assert hits and any("exponential backoff" in h.text for h in hits)


def test_chunk_chat_splits_long_turns():
    big = _line(type="user", message={"role": "user", "content": "word " * 800})  # ~4000 chars
    chunks = chunk_chat(big)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1500 for c in chunks)


def test_empty_or_noise_only_transcript():
    only_noise = "\n".join([_line(type="system", content="x"), _line(type="mode", mode="y")])
    assert chunk_chat(only_noise) == []


@pytest.fixture
def store(tmp_path):
    embedder = FakeEmbedder(dim=64)
    s = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    yield s, embedder, tmp_path
    s.close()


def test_index_transcript_as_kind_chat(store):
    s, embedder, tmp_path = store
    tp = tmp_path / "session.jsonl"
    tp.write_text(TRANSCRIPT)
    stats = index_transcript(tp, s, embedder)
    assert stats.files_indexed == 1
    assert stats.chunks_embedded == 3  # 3 clean turns
    # stored as chat, queryable by meaning, and filtered by kind
    [qvec] = embedder.embed(["retry with backoff"])
    hits = s.search_vec(qvec, k=5, kind="chat")
    assert hits and all(h.kind == "chat" for h in hits)
    assert any("exponential backoff" in h.text for h in hits)


def test_reindex_only_embeds_new_turns(store):
    s, embedder, tmp_path = store
    tp = tmp_path / "session.jsonl"
    tp.write_text(TRANSCRIPT)
    index_transcript(tp, s, embedder)
    # append a new turn (simulates the Stop hook firing on a grown transcript)
    new_turn = _line(
        type="assistant",
        message={
            "role": "assistant",
            "content": [{"type": "text", "text": "set a socket timeout"}],
        },
    )
    tp.write_text(TRANSCRIPT + "\n" + new_turn)
    stats = index_transcript(tp, s, embedder)
    assert stats.chunks_embedded == 1  # only the new turn re-embeds
    assert stats.chunks_reused == 3


def test_backfill_and_recall(store):
    s, embedder, tmp_path = store
    proj = tmp_path / "projects" / "some-project"
    proj.mkdir(parents=True)
    (proj / "a.jsonl").write_text(TRANSCRIPT)
    stats = backfill_chats(tmp_path / "projects", s, embedder)
    assert stats.files_indexed == 1

    from qmx.config import Settings

    settings = Settings(db_path=tmp_path / "index.db", embed_dim=64, embed_model="fake",
                        ollama_url="http://127.0.0.1:9")  # fmt: skip
    svc = QmxService(settings, embedder)
    hits = svc.recall("how to retry requests", k=5)
    assert hits and all(h["kind"] == "chat" for h in hits)
