"""AST-aware code chunking via tree-sitter.

Strategy: walk the file's top-level nodes. Emit one chunk per top-level definition
(function/class/…), splitting a definition into its nested members when it is larger than
``MAX_CHUNK_LINES``. When a definition is split, the code *between* its members (the class
signature, docstring, class-level attributes, blank gaps) is windowed rather than dropped — so no
source is lost. Regions between definitions (imports, constants) become sliding-window chunks too.
Unsupported languages or parse failures fall back to whole-file windows.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from qmx.store import Chunk

MAX_CHUNK_LINES = 120
WINDOW_LINES = 60
WINDOW_OVERLAP = 10

# File extension -> tree-sitter language name.
EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".scala": "scala",
    ".swift": "swift",
    ".kt": "kotlin",
}

# Node types that are worth a chunk of their own, per language. Kept small and conservative;
# anything not listed falls through to window chunking.
_DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
    },
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item", "mod_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "ruby": {"method", "class", "module"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "csharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition"},
    "swift": {"function_declaration", "class_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
}


def language_for_path(path: str | Path) -> str | None:
    """tree-sitter language name for a file, or ``None`` if unsupported."""
    return EXT_LANG.get(Path(path).suffix.lower())


def chunk_code(text: str, language: str | None) -> list[Chunk]:
    """Chunk source ``text`` written in ``language`` (a tree-sitter name)."""
    if not text.strip():
        return []
    def_types = _DEF_TYPES.get(language or "")
    if language is None or def_types is None:
        return _window_chunks(text, start_line=1)
    try:
        parser = get_parser(language)  # type: ignore[arg-type]
        tree = parser.parse(text.encode("utf-8"))
    except Exception:  # noqa: BLE001 — any parser failure -> safe window fallback
        return _window_chunks(text, start_line=1)

    lines = text.splitlines()
    chunks: list[Chunk] = []
    cursor = 0  # 0-based line of the next un-emitted region
    for node in tree.root_node.children:
        node_start = node.start_point[0]
        if node.type in def_types:
            if node_start > cursor:  # gap before this def -> window it
                chunks.extend(_window_chunks_from_lines(lines, cursor, node_start))
            chunks.extend(_chunks_for_definition(node, def_types, lines))
            cursor = node.end_point[0] + 1
    if cursor < len(lines):  # trailing region after the last def
        chunks.extend(_window_chunks_from_lines(lines, cursor, len(lines)))
    if not chunks:  # no recognised defs at all
        return _window_chunks(text, start_line=1)
    for i, chunk in enumerate(chunks):
        chunk.ord = i
    return chunks


def _chunks_for_definition(node: Node, def_types: set[str], lines: list[str]) -> list[Chunk]:
    """A definition node -> one chunk, or (when too large) its members plus windowed gaps.

    A large definition is split into its shallowest nested members (e.g. a class's methods); the
    regions *around and between* those members — the signature, docstring, class-level attributes,
    and blank gaps — are windowed so nothing is lost. Members that are themselves oversized recurse.
    """
    span = node.end_point[0] - node.start_point[0] + 1
    if span <= MAX_CHUNK_LINES:
        return [_chunk_from_node(node)]
    members = _shallow_members(node, def_types)
    if not members:
        return [_chunk_from_node(node)]

    chunks: list[Chunk] = []
    cursor = node.start_point[0]  # 0-based; covers the def header before the first member
    for member in members:
        member_start = member.start_point[0]
        if member_start > cursor:  # header / attributes / gap before this member -> window it
            chunks.extend(_window_chunks_from_lines(lines, cursor, member_start))
        chunks.extend(_chunks_for_definition(member, def_types, lines))
        cursor = member.end_point[0] + 1
    end = node.end_point[0] + 1
    if cursor < end:  # trailing code after the last member
        chunks.extend(_window_chunks_from_lines(lines, cursor, end))
    return chunks


def _shallow_members(node: Node, def_types: set[str]) -> list[Node]:
    """The shallowest def-type descendants of ``node`` (not descending into a matched member).

    Yields non-overlapping members in source order (e.g. a class's methods, not their own nested
    defs), so windowing the gaps between them can't double-cover a region.
    """
    members: list[Node] = []

    def walk(n: Node) -> None:
        for child in n.children:
            if child.type in def_types:
                members.append(child)
            else:
                walk(child)

    walk(node)
    members.sort(key=lambda m: m.start_point[0])
    return members


def _chunk_from_node(node: Node) -> Chunk:
    return Chunk(
        text=node.text.decode("utf-8", errors="replace"),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        symbol=_symbol_of(node),
    )


def _symbol_of(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", errors="replace")
    # decorated_definition (python) wraps the real def; look one level in.
    for child in node.children:
        got = child.child_by_field_name("name")
        if got is not None:
            return got.text.decode("utf-8", errors="replace")
    return None


def _window_chunks(text: str, start_line: int) -> list[Chunk]:
    return _window_chunks_from_lines(text.splitlines(), start_line - 1, len(text.splitlines()))


def _window_chunks_from_lines(lines: list[str], start: int, end: int) -> list[Chunk]:
    """Sliding-line-window chunks over ``lines[start:end]`` (0-based, end-exclusive)."""
    chunks: list[Chunk] = []
    i = start
    step = max(1, WINDOW_LINES - WINDOW_OVERLAP)
    while i < end:
        j = min(i + WINDOW_LINES, end)
        body = "\n".join(lines[i:j]).strip()
        if body:
            chunks.append(Chunk(text=body, start_line=i + 1, end_line=j))
        if j >= end:
            break
        i += step
    return chunks
