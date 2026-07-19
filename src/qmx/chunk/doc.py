"""Markdown chunking — small, heading-aware. Used for repo docs (``kind=doc``) and Claude memory.

Splits on ATX headings (`#`..`######`): each heading + its body becomes one chunk (the heading is
kept as ``symbol`` and prepended to the text for context); content before the first heading is its
own chunk. Over-long sections are split by size. No code-fence parsing yet — a `#`-comment inside a
fenced block is treated as a heading — good enough for the short, prose-y ``*.md`` files indexed.
"""

from __future__ import annotations

import re

from qmx.store import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MAX_CHARS = 1500


def chunk_markdown(text: str) -> list[Chunk]:
    """Chunk markdown ``text`` into heading-delimited sections."""
    lines = text.splitlines()
    sections: list[tuple[int, str | None, list[str]]] = []  # (start_line, heading, body_lines)
    current: tuple[int, str | None, list[str]] = (1, None, [])
    for i, line in enumerate(lines, 1):
        m = _HEADING.match(line)
        if m:
            if current[2] or current[1] is not None:
                sections.append(current)
            current = (i, m.group(2), [line])
        else:
            current[2].append(line)
    if current[2] or current[1] is not None:
        sections.append(current)

    chunks: list[Chunk] = []
    for start_line, heading, body in sections:
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        for piece in _split(body_text):
            chunks.append(Chunk(text=piece, ord=len(chunks), symbol=heading, start_line=start_line))
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
        while len(buf) > _MAX_CHARS:
            pieces.append(buf[:_MAX_CHARS])
            buf = buf[_MAX_CHARS:]
    if buf.strip():
        pieces.append(buf.strip())
    return pieces
