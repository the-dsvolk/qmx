"""Meeting chunking — Memoir call folders → ``kind=meeting`` chunks.

A Memoir call folder (``<save dir>/YYYY-MM-DD_<name>/``) holds ``transcript.txt``, ``summary.md``,
``notes.md``, ``prep.md`` and ``meeting.json``. Each indexed file becomes its own document; every
chunk starts with a one-line header — ``name · date [start] · attendees`` — so a search for the
meeting, the day or a person matches even when the chunk body never mentions them.

The transcript is chunked from ``transcript.txt``: Memoir writes it with late corrections already
applied (the last version of each segment) and sorted by start time, one ``[mm:ss] Speaker: text``
line per segment. Lines are grouped into windows of about 60–90 seconds (or ~1,500 characters),
breaking only between lines, with the last line of a window repeated at the start of the next.
Each chunk cites its ``transcript.txt`` line range and carries ``Speakers mm:ss–mm:ss`` as its
``symbol``. Markdown files go through :func:`~qmx.chunk.doc.chunk_markdown` and get the header too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qmx.chunk.doc import chunk_markdown
from qmx.store import Chunk

WINDOW_MIN_SECONDS = 60
WINDOW_MAX_SECONDS = 90
MAX_CHARS = 1500

_LINE = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\]\s+([^:]+):\s?(.*)$")
_FOLDER = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")
_ATTENDEES = re.compile(r"^\s*(?:[-*]\s+)?attendees\s*:\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MeetingInfo:
    """What a call's chunk header says about it."""

    name: str
    date: str = ""
    start: str = ""
    attendees: tuple[str, ...] = field(default_factory=tuple)

    def header(self) -> str:
        when = f"{self.date} {self.start}".strip()
        parts = [self.name, when, ", ".join(self.attendees)]
        return " · ".join(p for p in parts if p)


def read_meeting_info(folder: Path) -> MeetingInfo:
    """Name, date, start time and attendees for a call folder.

    Name and start come from ``meeting.json`` (``name``, ``started_at``), falling back to the
    folder name ``YYYY-MM-DD_<slug>``. Attendees come from an ``Attendees:`` line in ``prep.md``.
    """
    folder = Path(folder)
    meta: dict = {}
    meta_path = folder / "meeting.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}

    m = _FOLDER.match(folder.name)
    name = str(meta.get("name") or (m.group(2).replace("-", " ") if m else folder.name))
    date = m.group(1) if m else ""
    start = ""
    started_at = meta.get("started_at")
    if isinstance(started_at, str):
        try:
            started = datetime.fromisoformat(started_at)
            date = started.strftime("%Y-%m-%d")
            start = started.strftime("%H:%M")
        except ValueError:
            pass

    prep = folder / "prep.md"
    attendees = parse_attendees(prep.read_text(encoding="utf-8")) if prep.is_file() else ()
    return MeetingInfo(name=name, date=date, start=start, attendees=attendees)


def parse_attendees(prep_text: str) -> tuple[str, ...]:
    """Names from the first ``Attendees: a, b, c`` line (``-``/``*`` bullets allowed)."""
    for line in prep_text.splitlines():
        m = _ATTENDEES.match(line)
        if m:
            names = re.split(r"[,;]", m.group(1))
            return tuple(n.strip() for n in names if n.strip())
    return ()


@dataclass(slots=True)
class _Line:
    number: int  # 1-based line in transcript.txt
    seconds: int
    speaker: str
    text: str


def _parse_transcript(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if m:
            hours, minutes, secs, speaker, body = m.groups()
            seconds = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
            lines.append(_Line(number, seconds, speaker.strip(), raw))
        elif lines:  # not a segment line: keep it with the one before (same time)
            prev = lines[-1]
            lines.append(_Line(number, prev.seconds, prev.speaker, raw))
    return lines


def _clock(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _window_chunk(window: list[_Line], header: str, ord_: int) -> Chunk:
    speakers = list(dict.fromkeys(line.speaker for line in window))
    span = f"{_clock(window[0].seconds)}–{_clock(window[-1].seconds)}"
    body = "\n".join(line.text for line in window)
    return Chunk(
        text=f"{header}\n{body}" if header else body,
        ord=ord_,
        start_line=window[0].number,
        end_line=window[-1].number,
        symbol=f"{'/'.join(speakers)} {span}",
    )


def chunk_transcript(text: str, header: str = "") -> list[Chunk]:
    """Chunk ``transcript.txt`` into ~60–90 s windows with a one-line overlap."""
    lines = _parse_transcript(text)
    chunks: list[Chunk] = []
    window: list[_Line] = []
    chars = 0

    def flush() -> None:
        nonlocal window, chars
        chunks.append(_window_chunk(window, header, len(chunks)))
        carry = window[-1:] if len(window) > 1 else []
        window = list(carry)
        chars = sum(len(line.text) + 1 for line in window)

    for line in lines:
        too_long = window and line.seconds - window[0].seconds > WINDOW_MAX_SECONDS
        too_big = window and chars + len(line.text) + 1 > MAX_CHARS
        if too_long or too_big:
            flush()
        window.append(line)
        chars += len(line.text) + 1
        if line.seconds - window[0].seconds >= WINDOW_MIN_SECONDS or chars >= MAX_CHARS:
            flush()

    if window and (not chunks or window[-1].number > (chunks[-1].end_line or 0)):
        chunks.append(_window_chunk(window, header, len(chunks)))
    return chunks


def chunk_meeting_markdown(text: str, header: str = "") -> list[Chunk]:
    """Chunk a call's markdown (summary / notes / prep), prefixing every chunk with ``header``."""
    if not header:
        return chunk_markdown(text)
    return [
        Chunk(
            text=f"{header}\n{c.text}",
            ord=c.ord,
            start_line=c.start_line,
            end_line=c.end_line,
            symbol=c.symbol,
        )
        for c in chunk_markdown(text)
    ]
