"""Indexer — walk source trees, chunk code, embed, and upsert into the store.

Phase 1 scope: a clean code-only walk with a per-file ``file_hash`` skip for unchanged files and a
wholesale re-chunk of changed ones. The heavier per-chunk diffing, tombstones on delete, and a
filesystem watcher are the Phase 2 robustness core (``plan/qmx-plan.md``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from qmx.chunk.code import chunk_code, language_for_path
from qmx.embed import Embedder
from qmx.store import Store, hash_text

log = logging.getLogger("qmx.index")

# Directories never worth indexing (pruned during the walk).
EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "target",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "site-packages",
        ".idea",
        ".vscode",
        ".qmx",
    }
)
MAX_FILE_BYTES = 1_000_000  # skip files larger than ~1 MB (logged, not silent)


@dataclass(slots=True)
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: IndexStats) -> None:
        self.files_scanned += other.files_scanned
        self.files_indexed += other.files_indexed
        self.files_skipped += other.files_skipped
        self.chunks_added += other.chunks_added
        self.chunks_removed += other.chunks_removed
        self.errors.extend(other.errors)


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield indexable code files under ``root`` (a dir or a single file), pruning junk dirs."""
    if root.is_file():
        if language_for_path(root) is not None:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if language_for_path(path) is not None:
                yield path


def index_paths(
    paths: list[Path], store: Store, embedder: Embedder, *, force: bool = False
) -> IndexStats:
    """Index code files under ``paths``. Unchanged files (by hash) are skipped unless ``force``."""
    stats = IndexStats()
    for root in paths:
        root = root.resolve()
        repo = root.name if root.is_dir() else root.parent.name
        for file_path in iter_source_files(root):
            stats.files_scanned += 1
            try:
                _index_file(file_path, repo, store, embedder, force, stats)
            except OSError as exc:
                stats.errors.append(f"{file_path}: {exc}")
                log.warning("skip %s: %s", file_path, exc)
    return stats


def _index_file(
    file_path: Path, repo: str, store: Store, embedder: Embedder, force: bool, stats: IndexStats
) -> None:
    size = file_path.stat().st_size
    if size > MAX_FILE_BYTES:
        stats.files_skipped += 1
        log.info("skip oversized %s (%d bytes)", file_path, size)
        return

    text = file_path.read_text(encoding="utf-8", errors="replace")
    path_key = str(file_path)
    file_hash = hash_text(text)
    if not force and store.document_hash("code", path_key) == file_hash:
        stats.files_skipped += 1
        return

    chunks = chunk_code(text, language_for_path(file_path))
    # Embed BEFORE any DB write: if the backend is down this raises and we leave no document row,
    # so the file's file_hash is never persisted and a later run re-processes it (not silently
    # skipped as "unchanged"). Files indexed earlier in the run stay committed -> resumable.
    embeddings = embedder.embed([c.text for c in chunks]) if chunks else []

    doc_id = store.upsert_document(
        kind="code", path=path_key, repo=repo, mtime=file_path.stat().st_mtime, file_hash=file_hash
    )
    stats.chunks_removed += store.clear_document_chunks(doc_id)
    if chunks:
        store.add_chunks(doc_id, chunks, embeddings)
        stats.chunks_added += len(chunks)
    stats.files_indexed += 1
