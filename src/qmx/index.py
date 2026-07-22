"""Indexer — walk source trees, chunk code, embed only what changed, and upsert into the store.

Phase 2 robustness core (``plan/qmx-plan.md``):

- **Incremental:** unchanged files skip on ``file_hash``; a changed file re-embeds only its
  new/edited chunks (per-chunk hash diff), reusing everything else.
- **Deletes:** a directory re-scan tombstones documents that vanished from disk.
- **Crash-safe:** each file embeds *before* any DB write, so a backend failure leaves no
  half-written document and files done earlier stay committed (resumable). The document's
  ``file_hash`` (the "fully indexed" marker) is recorded **only after** its chunks/mentions are
  committed — so an index interrupted mid-write is re-processed next run, never silently skipped
  as "already indexed" with no chunks.
"""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qmx.chunk.chat import ChatSource, chunk_chat
from qmx.chunk.code import chunk_code, language_for_path
from qmx.chunk.doc import chunk_markdown
from qmx.embed import Embedder
from qmx.store import Chunk, ReindexResult, Store, hash_text

log = logging.getLogger("qmx.index")

# Directories never worth indexing (pruned during the walk).
EXCLUDE_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "dist", "build", "target",
        ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".tox", "site-packages", ".idea", ".vscode", ".qmx",
    }
)  # fmt: skip
MAX_FILE_BYTES = 1_000_000  # skip files larger than ~1 MB (logged, not silent)

# Markdown docs indexed from repos as kind="doc" (chunked by chunk_markdown, not tree-sitter).
DOC_EXTS = frozenset({".md", ".markdown"})


def repo_kind(path: Path) -> str | None:
    """Index kind for a repo file: ``code`` (tree-sitter), ``doc`` (markdown), or ``None``."""
    if language_for_path(path) is not None:
        return "code"
    if Path(path).suffix.lower() in DOC_EXTS:
        return "doc"
    return None


@dataclass(slots=True)
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_added: int = 0  # mentions written across indexed files
    chunks_embedded: int = 0  # content that actually required an embedding call
    chunks_reused: int = 0  # content served from dedup / unchanged / revived tombstone
    chunks_orphaned: int = 0  # chunks tombstoned by edits/deletes this run
    errors: list[str] = field(default_factory=list)


def embed_missing(
    store: Store, embedder: Embedder, chunks: Sequence[Chunk]
) -> dict[str, list[float]]:
    """Embed only the chunk hashes not already in ``store``; returns ``{hash: vector}``."""
    missing = list(store.missing_chunk_hashes({c.chunk_hash for c in chunks}))
    if not missing:
        return {}
    text_by_hash = {c.chunk_hash: c.text for c in chunks}
    vectors = embedder.embed([text_by_hash[h] for h in missing])
    return dict(zip(missing, vectors, strict=True))


def reindex(
    store: Store, embedder: Embedder, doc_id: int, chunks: Sequence[Chunk]
) -> ReindexResult:
    """Embed missing content then replace ``doc_id``'s mentions. Embeds before any DB write."""
    new_embeddings = embed_missing(store, embedder, chunks)
    return store.reindex_document(doc_id, chunks, new_embeddings)


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield indexable files (code + markdown) under ``root``, pruning junk dirs."""
    if root.is_file():
        if repo_kind(root) is not None:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if repo_kind(path) is not None:
                yield path


def index_paths(
    paths: list[Path], store: Store, embedder: Embedder, *, force: bool = False
) -> IndexStats:
    """Index code files under ``paths``; prune deleted files under directory roots."""
    stats = IndexStats()
    for raw_root in paths:
        root = raw_root.resolve()
        repo = root.name if root.is_dir() else root.parent.name
        seen: set[str] = set()
        for file_path in iter_source_files(root):
            stats.files_scanned += 1
            seen.add(str(file_path))
            try:
                _index_file(file_path, repo, store, embedder, force, stats)
            except OSError as exc:
                stats.errors.append(f"{file_path}: {exc}")
                log.warning("skip %s: %s", file_path, exc)
        if root.is_dir():
            _prune_deleted(root, seen, store, stats)
    return stats


def _index_file(
    file_path: Path, repo: str, store: Store, embedder: Embedder, force: bool, stats: IndexStats
) -> None:
    size = file_path.stat().st_size
    if size > MAX_FILE_BYTES:
        stats.files_skipped += 1
        log.info("skip oversized %s (%d bytes)", file_path, size)
        return

    kind = repo_kind(file_path)
    if kind is None:
        return

    text = file_path.read_text(encoding="utf-8", errors="replace")
    path_key = str(file_path)
    file_hash = hash_text(text)
    if not force and store.document_hash(kind, path_key) == file_hash:
        stats.files_skipped += 1
        return

    # Route by kind: tree-sitter for code, markdown chunker for docs.
    chunks = (
        chunk_code(text, language_for_path(file_path)) if kind == "code" else chunk_markdown(text)
    )
    # Embed missing content BEFORE writing the document, so a backend failure persists nothing
    # (the file's file_hash is never recorded -> a later run re-processes it, not silently skipped).
    new_embeddings = embed_missing(store, embedder, chunks)

    # Upsert without the hash, write chunks/mentions, THEN stamp file_hash: the hash is the
    # "fully indexed" marker, so an interrupted write is re-processed rather than silently skipped.
    doc_id = store.upsert_document(
        kind=kind, path=path_key, repo=repo, mtime=file_path.stat().st_mtime
    )
    result = store.reindex_document(doc_id, chunks, new_embeddings)
    store.set_document_hash(doc_id, file_hash)

    stats.files_indexed += 1
    stats.chunks_added += result.mentions
    stats.chunks_embedded += result.embedded
    stats.chunks_reused += result.reused
    stats.chunks_orphaned += result.orphaned


def _prune_deleted(root: Path, seen: set[str], store: Store, stats: IndexStats) -> None:
    prefix = str(root) + os.sep
    for kind in ("code", "doc"):
        for doc_id, path in store.documents_under(kind, prefix):
            if path not in seen:
                stats.chunks_orphaned += store.remove_document_by_id(doc_id)
                stats.files_removed += 1
                log.info("removed deleted file %s", path)


# -- chats (kind="chat") -----------------------------------------------------------------------


def index_transcript(
    path: Path,
    store: Store,
    embedder: Embedder,
    *,
    force: bool = False,
    source: ChatSource = "claude",
) -> IndexStats:
    """Index one JSONL transcript as ``kind='chat'`` (used by backfill + capture).

    ``source`` selects the transcript schema (``"claude"`` or ``"cursor"``). Cheap on re-runs: the
    whole transcript is re-chunked, but the per-chunk dedup means only *new* turns embed — so the
    Stop hook re-indexing a growing file only pays for the latest turn(s).
    """
    stats = IndexStats()
    stats.files_scanned += 1
    _ingest_transcript(Path(path), store, embedder, force, stats, source)
    return stats


# Nested transcript dirs to skip: subagent + workflow sub-transcripts are internal machinery,
# not the human/assistant conversation (they'd add empty/noise docs and pollute recall).
_CHAT_SKIP_DIRS = frozenset({"subagents", "workflows"})


def backfill_chats(
    projects_dir: Path,
    store: Store,
    embedder: Embedder,
    *,
    force: bool = False,
    source: ChatSource = "claude",
) -> IndexStats:
    """Index the main session transcripts under ``projects_dir`` (e.g. ``~/.claude/projects``).

    ``source`` selects the transcript schema (``"claude"`` or ``"cursor"``). Only top-level
    ``<project>/<session>.jsonl`` files — subagent/workflow sub-transcripts (under ``subagents/`` or
    ``workflows/``) are skipped as internal machinery.
    """
    stats = IndexStats()
    for jsonl in sorted(Path(projects_dir).rglob("*.jsonl")):
        if _CHAT_SKIP_DIRS.intersection(jsonl.parts):
            continue
        stats.files_scanned += 1
        try:
            _ingest_transcript(jsonl, store, embedder, force, stats, source)
        except OSError as exc:
            stats.errors.append(f"{jsonl}: {exc}")
            log.warning("skip %s: %s", jsonl, exc)
    return stats


def _ingest_transcript(
    path: Path,
    store: Store,
    embedder: Embedder,
    force: bool,
    stats: IndexStats,
    source: ChatSource = "claude",
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    path_key = str(path.resolve())
    file_hash = hash_text(text)
    if not force and store.document_hash("chat", path_key) == file_hash:
        stats.files_skipped += 1
        return

    chunks = chunk_chat(text, source)
    new_embeddings = embed_missing(store, embedder, chunks)  # embed before any DB write
    # file_hash last (after chunks land) so an interrupted write is re-processed, not skipped.
    doc_id = store.upsert_document(
        kind="chat",
        path=path_key,
        repo=path.parent.name,
        mtime=path.stat().st_mtime,
    )
    result = store.reindex_document(doc_id, chunks, new_embeddings)
    store.set_document_hash(doc_id, file_hash)

    stats.files_indexed += 1
    stats.chunks_added += result.mentions
    stats.chunks_embedded += result.embedded
    stats.chunks_reused += result.reused
    stats.chunks_orphaned += result.orphaned


# -- memory (kind="memory") --------------------------------------------------------------------


def index_memory(
    globs: Iterable[str], store: Store, embedder: Embedder, *, force: bool = False
) -> IndexStats:
    """Index Claude memory markdown from ``globs`` (``~`` expanded) as ``kind='memory'``.

    A glob matching a directory is scanned recursively for ``*.md``; a glob matching a ``.md`` file
    is taken directly. See ``Settings.memory_globs`` (default: per-project ``memory/`` dirs).
    """
    stats = IndexStats()
    for md in _iter_memory_files(globs):
        stats.files_scanned += 1
        try:
            _ingest_markdown(md, store, embedder, force, stats)
        except OSError as exc:
            stats.errors.append(f"{md}: {exc}")
            log.warning("skip %s: %s", md, exc)
    return stats


def index_memory_dir(
    memory_dir: Path, store: Store, embedder: Embedder, *, force: bool = False
) -> IndexStats:
    """Index every ``*.md`` under one memory directory (used by capture for a session's sibling)."""
    stats = IndexStats()
    if not Path(memory_dir).is_dir():
        return stats
    for md in sorted(Path(memory_dir).rglob("*.md")):
        stats.files_scanned += 1
        _ingest_markdown(md, store, embedder, force, stats)
    return stats


def _iter_memory_files(globs: Iterable[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for match in sorted(glob.glob(os.path.expanduser(pattern))):
            p = Path(match)
            if p.is_dir():
                candidates = sorted(p.rglob("*.md"))
            elif p.suffix == ".md":
                candidates = [p]
            else:
                candidates = []
            for md in candidates:
                if md not in seen:
                    seen.add(md)
                    yield md


def _ingest_markdown(
    path: Path, store: Store, embedder: Embedder, force: bool, stats: IndexStats
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    path_key = str(path.resolve())
    file_hash = hash_text(text)
    if not force and store.document_hash("memory", path_key) == file_hash:
        stats.files_skipped += 1
        return

    chunks = chunk_markdown(text)
    new_embeddings = embed_missing(store, embedder, chunks)
    # file_hash last (after chunks land) so an interrupted write is re-processed, not skipped.
    doc_id = store.upsert_document(
        kind="memory",
        path=path_key,
        repo=path.parent.name,
        mtime=path.stat().st_mtime,
    )
    result = store.reindex_document(doc_id, chunks, new_embeddings)
    store.set_document_hash(doc_id, file_hash)

    stats.files_indexed += 1
    stats.chunks_added += result.mentions
    stats.chunks_embedded += result.embedded
    stats.chunks_reused += result.reused
    stats.chunks_orphaned += result.orphaned
