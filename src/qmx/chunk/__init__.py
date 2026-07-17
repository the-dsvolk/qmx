"""Chunkers — turn source files into indexable :class:`~qmx.store.Chunk` units.

``code`` (tree-sitter AST) and ``chat`` (Claude Code JSONL transcripts). ``doc`` (markdown) follows
per ``plan/qmx-plan.md``.
"""

from qmx.chunk.chat import chunk_chat
from qmx.chunk.code import chunk_code, language_for_path
from qmx.chunk.doc import chunk_markdown

__all__ = ["chunk_chat", "chunk_code", "chunk_markdown", "language_for_path"]
