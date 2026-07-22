"""Chat transcript chunking — Claude Code / Cursor JSONL → clean conversation turns.

Reads a JSONL transcript directly (no intermediate markdown) and keeps only human/assistant
**text**: `thinking`, `tool_use`, `tool_result`, and `image` blocks are dropped, as are the
non-message line types and subagent side-chains. Each surviving turn becomes one chunk (long turns
split by size), stamped with its role (`symbol`) and JSONL line number (`start_line`) so results
cite a spot in the transcript.

Two on-disk schemas are supported, selected by an explicit ``source`` flag (no per-line sniffing):

* ``"claude"`` — ``~/.claude/projects/*/*.jsonl``. Turn lines carry a top-level ``type`` of
  ``user``/``assistant``, an ``isSidechain`` flag, and a nested ``message`` dict whose ``role``
  names the speaker. Non-message lines (``system``, ``mode``, ``file-history-snapshot``, …) exist.
* ``"cursor"`` — ``~/.cursor/projects/*/agent-transcripts/<uuid>/<uuid>.jsonl``. Turn lines carry a
  top-level ``role`` of ``user``/``assistant`` and a ``message`` dict that has **no inner role**;
  marker lines are ``{"type": "turn_ended", ...}`` (no ``role``). No ``tool_result``/``thinking``
  blocks are emitted. See the format-verification notes in the port research.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from qmx.store import Chunk

ChatSource = Literal["claude", "cursor"]

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


def _turn_claude(obj: dict) -> tuple[str, dict] | None:
    """Claude line → ``(role, message)`` for a real turn, or ``None`` to skip.

    Turns carry a top-level ``type`` of user/assistant; sidechains and non-message line types are
    dropped. The speaker is ``message.role`` (falling back to the top-level ``type``).
    """
    if obj.get("type") not in _KEEP_TYPES or obj.get("isSidechain"):
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    return (msg.get("role") or obj["type"]), msg


def _turn_cursor(obj: dict) -> tuple[str, dict] | None:
    """Cursor line → ``(role, message)`` for a real turn, or ``None`` to skip.

    The speaker is the top-level ``role``; ``turn_ended`` marker lines (no ``role``) are dropped and
    the ``message`` dict has no inner role.
    """
    role = obj.get("role")
    if role not in _KEEP_TYPES:
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    return role, msg


_TURN_PARSERS = {"claude": _turn_claude, "cursor": _turn_cursor}


def iter_turns(jsonl_text: str, source: ChatSource = "claude"):
    """Yield ``(line_no, role, text)`` for each real human/assistant turn, in order.

    ``source`` selects the line schema (``"claude"`` or ``"cursor"``); the ``content``-block shape
    and text cleaning are shared across both.
    """
    parse = _TURN_PARSERS[source]
    for line_no, raw in enumerate(jsonl_text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed = parse(obj)
        if parsed is None:
            continue
        role, msg = parsed
        text = _clean_text(msg.get("content"))
        if len(text) < 2:  # e.g. a turn that was only a tool_result / thinking → nothing left
            continue
        yield line_no, role, text


def chunk_chat(jsonl_text: str, source: ChatSource = "claude") -> list[Chunk]:
    """Chunk a transcript's text into per-turn :class:`~qmx.store.Chunk`s."""
    chunks: list[Chunk] = []
    for line_no, role, text in iter_turns(jsonl_text, source):
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
