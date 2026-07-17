"""Filesystem watcher — keep the index live as code changes.

Wraps ``watchdog``: on create/modify of a code file, incrementally reindex it; on delete, tombstone
its document; on move, do both. The routing lives in :class:`CodeChangeHandler` (unit-tested with
synthetic events); :func:`watch` wires it to an OS observer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from qmx.chunk.code import language_for_path
from qmx.embed import Embedder
from qmx.index import index_paths
from qmx.store import Store

log = logging.getLogger("qmx.watch")


class CodeChangeHandler(FileSystemEventHandler):
    """Translates watchdog events into incremental (re)index / remove calls."""

    def __init__(self, store: Store, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @staticmethod
    def _is_code(path: str) -> bool:
        return bool(path) and language_for_path(path) is not None

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_code(event.src_path):
            self._reindex(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_code(event.src_path):
            self._reindex(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_code(event.src_path):
            self._remove(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._is_code(event.src_path):
            self._remove(event.src_path)
        dest = getattr(event, "dest_path", "")
        if self._is_code(dest):
            self._reindex(dest)

    def _reindex(self, path: str) -> None:
        try:
            index_paths([Path(path)], self._store, self._embedder)
            log.info("reindexed %s", path)
        except Exception as exc:  # noqa: BLE001 — a watcher must not die on one bad file
            log.warning("reindex failed for %s: %s", path, exc)

    def _remove(self, path: str) -> None:
        self._store.remove_document("code", str(Path(path).resolve()))
        log.info("removed %s", path)


def watch(paths: list[Path], store: Store, embedder: Embedder, *, block: bool = True) -> Observer:
    """Watch ``paths`` recursively and keep the index in sync. Blocks until interrupted."""
    observer = Observer()
    handler = CodeChangeHandler(store, embedder)
    for p in paths:
        observer.schedule(handler, str(Path(p).resolve()), recursive=True)
    observer.start()
    log.info("watching %s (Ctrl-C to stop)", ", ".join(str(p) for p in paths))
    if block:
        try:
            while observer.is_alive():
                observer.join(1)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
    return observer
