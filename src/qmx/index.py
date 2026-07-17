"""Indexer — walk source trees, chunk code, embed only what changed, and upsert into the store.

Phase 2 robustness core (``plan/qmx-plan.md``):

- **Incremental:** unchanged files skip on ``file_hash``; a changed file re-embeds only its
  new/edited chunks (per-chunk hash diff), reusing everything else.
- **Deletes:** a directory re-scan tombstones documents that vanished from disk.
- **Crash-safe:** each file embeds *before* any DB write, so a backend failure leaves no
  half-written document and files done earlier stay committed (resumable).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qmx.chunk.code import chunk_code, language_for_path
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

    text = file_path.read_text(encoding="utf-8", errors="replace")
    path_key = str(file_path)
    file_hash = hash_text(text)
    if not force and store.document_hash("code", path_key) == file_hash:
        stats.files_skipped += 1
        return

    chunks = chunk_code(text, language_for_path(file_path))
    # Embed missing content BEFORE writing the document, so a backend failure persists nothing
    # (the file's file_hash is never recorded -> a later run re-processes it, not silently skipped).
    new_embeddings = embed_missing(store, embedder, chunks)

    doc_id = store.upsert_document(
        kind="code", path=path_key, repo=repo, mtime=file_path.stat().st_mtime, file_hash=file_hash
    )
    result = store.reindex_document(doc_id, chunks, new_embeddings)

    stats.files_indexed += 1
    stats.chunks_added += result.mentions
    stats.chunks_embedded += result.embedded
    stats.chunks_reused += result.reused
    stats.chunks_orphaned += result.orphaned


def _prune_deleted(root: Path, seen: set[str], store: Store, stats: IndexStats) -> None:
    prefix = str(root) + os.sep
    for doc_id, path in store.documents_under("code", prefix):
        if path not in seen:
            stats.chunks_orphaned += store.remove_document_by_id(doc_id)
            stats.files_removed += 1
            log.info("removed deleted file %s", path)
