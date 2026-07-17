"""Chunkers — turn source files into indexable :class:`~qmx.store.Chunk` units.

Phase 1 ships ``code`` (tree-sitter AST). ``doc`` (markdown) and ``chat`` (jsonl) follow in later
phases per ``plan/qmx-plan.md``.
"""

from qmx.chunk.code import chunk_code, language_for_path

__all__ = ["chunk_code", "language_for_path"]
