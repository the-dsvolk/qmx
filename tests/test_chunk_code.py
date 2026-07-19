"""tree-sitter code chunking: symbols, line spans, splitting, and fallbacks."""

from __future__ import annotations

from qmx.chunk.code import MAX_CHUNK_LINES, chunk_code, language_for_path

PY = '''\
import os


def alpha(x):
    """First."""
    return x + 1


class Widget:
    def method_one(self):
        return 1

    def method_two(self):
        return 2
'''


def test_language_for_path():
    assert language_for_path("a/b/foo.py") == "python"
    assert language_for_path("foo.rs") == "rust"
    assert language_for_path("foo.unknownext") is None


def test_python_defs_become_chunks_with_symbols_and_lines():
    chunks = chunk_code(PY, "python")
    symbols = {c.symbol for c in chunks}
    assert "alpha" in symbols
    assert "Widget" in symbols  # small class stays whole
    alpha = next(c for c in chunks if c.symbol == "alpha")
    assert alpha.start_line == 4
    assert "return x + 1" in alpha.text
    assert list(range(len(chunks))) == [c.ord for c in chunks]  # ords are sequential


def test_large_class_splits_into_methods():
    filler = "\n".join(f"        v{i} = {i}" for i in range(MAX_CHUNK_LINES + 10))
    big = f"class Big:\n    def only(self):\n{filler}\n        return 0\n"
    chunks = chunk_code(big, "python")
    assert "only" in {c.symbol for c in chunks}  # split to the method, not the whole class


def test_large_class_preserves_header_and_attributes():
    # A class bigger than MAX_CHUNK_LINES with a docstring + class attribute before the methods.
    filler = "\n".join(f"        step_{i}()" for i in range(MAX_CHUNK_LINES + 10))
    big = (
        "class Big:\n"
        '    """Big docstring."""\n'
        "    KIND = 'widget'\n"
        "    def only(self):\n"
        f"{filler}\n"
        "        return 0\n"
    )
    chunks = chunk_code(big, "python")
    symbols = {c.symbol for c in chunks}
    joined = "\n".join(c.text for c in chunks)
    assert "only" in symbols  # the method is still its own AST-aligned chunk
    # ...and the header/docstring/attribute are windowed, not dropped (the bug this fixes).
    assert "class Big:" in joined
    assert "Big docstring." in joined
    assert "KIND = 'widget'" in joined


def test_top_level_code_is_windowed_not_lost():
    src = "import sys\nCONST = 42\n\n\ndef f():\n    return CONST\n"
    chunks = chunk_code(src, "python")
    joined = "\n".join(c.text for c in chunks)
    assert "CONST = 42" in joined  # module-level statements still indexed
    assert any(c.symbol == "f" for c in chunks)


def test_unknown_language_falls_back_to_windows():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_code(text, None)
    assert len(chunks) > 1
    assert all(c.symbol is None for c in chunks)


def test_empty_text_yields_nothing():
    assert chunk_code("   \n  \n", "python") == []
