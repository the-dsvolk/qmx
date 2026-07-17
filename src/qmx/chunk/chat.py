"""Chat transcript chunking — Claude Code JSONL → clean conversation turns.

Reads a `~/.claude/projects/*/*.jsonl` transcript directly (no intermediate markdown) and keeps only
human/assistant **text**: `thinking`, `tool_use`, `tool_result`, and `image` blocks are dropped, as
are the non-message line types (`system`, `mode`, `queue-operation`, `file-history-snapshot`, …) and
subagent side-chains. Each surviving turn becomes one chunk (long turns split by size), stamped with
its role (`symbol`) and JSONL line number (`start_line`) so results cite a spot in the transcript.
"""

from __future__ import annotations

import json
import re

from qmx.store import Chunk

_KEEP_TYPES = {"user", "assistant"}
_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_MAX_CHARS = 1500  # split turns longer than this into multiple chunks


def _clean_text(content: object) -> str:
    """Extract human-readable text from a message's ``content`` (string or block list)."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    else:
        return ""
    return _SYSTEM_REMINDER.sub("", text).strip()


def iter_turns(jsonl_text: str):
    """Yield ``(line_no, role, text)`` for each real human/assistant turn, in order."""
    for line_no, raw in enumerate(jsonl_text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") not in _KEEP_TYPES or obj.get("isSidechain"):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        text = _clean_text(msg.get("content"))
        if len(text) < 2:  # e.g. a turn that was only a tool_result / thinking → nothing left
            continue
        yield line_no, (msg.get("role") or obj["type"]), text


def chunk_chat(jsonl_text: str) -> list[Chunk]:
    """Chunk a transcript's text into per-turn :class:`~qmx.store.Chunk`s."""
    chunks: list[Chunk] = []
    for line_no, role, text in iter_turns(jsonl_text):
        for piece in _split(text):
            chunks.append(Chunk(text=piece, ord=len(chunks), symbol=role, start_line=line_no))
    return chunks


def _split(text: str) -> list[str]:
    if len(text) <= _MAX_CHARS:
        return [text]
    pieces: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) + 2 > _MAX_CHARS:
            pieces.append(buf.strip())
            buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
        while len(buf) > _MAX_CHARS:  # a single oversized paragraph
            pieces.append(buf[:_MAX_CHARS])
            buf = buf[_MAX_CHARS:]
    if buf.strip():
        pieces.append(buf.strip())
    return pieces
