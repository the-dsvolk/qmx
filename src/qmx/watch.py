"""Filesystem watcher — keep the index live as code changes.

Wraps ``watchdog``: on create/modify of a code file, incrementally reindex it; on delete, tombstone
its document; on move, do both. The routing lives in :class:`CodeChangeHandler` (unit-tested with
synthetic events); :func:`watch` wires it to an OS observer.

Threading: watchdog calls its handlers on the observer's own thread, but the store's ``sqlite3``
connection may only be used on the thread that opened it. So the observer is given an
:class:`EventRelay`, which only queues events; :func:`watch` replays them into the
:class:`CodeChangeHandler` on the thread that owns the store, one at a time. That also keeps this
process to a single writer connection (other qmx processes contend through WAL + busy_timeout).
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from qmx.embed import Embedder
from qmx.index import index_paths, repo_kind
from qmx.store import Store

log = logging.getLogger("qmx.watch")


class CodeChangeHandler(FileSystemEventHandler):
    """Translates watchdog events into incremental (re)index / remove calls.

    Does the store work synchronously, so call it only on the thread that opened ``store`` —
    :func:`watch` feeds it through an :class:`EventRelay` rather than scheduling it on the
    observer directly.
    """

    def __init__(self, store: Store, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @staticmethod
    def _indexable(path: str) -> bool:
        return bool(path) and repo_kind(Path(path)) is not None

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._indexable(event.src_path):
            self._reindex(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._indexable(event.src_path):
            self._reindex(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._indexable(event.src_path):
            self._remove(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._indexable(event.src_path):
            self._remove(event.src_path)
        dest = getattr(event, "dest_path", "")
        if self._indexable(dest):
            self._reindex(dest)

    def _reindex(self, path: str) -> None:
        try:
            index_paths([Path(path)], self._store, self._embedder)
            log.info("reindexed %s", path)
        except Exception as exc:  # noqa: BLE001 — a watcher must not die on one bad file
            log.warning("reindex failed for %s: %s", path, exc)

    def _remove(self, path: str) -> None:
        kind = repo_kind(Path(path)) or "code"
        try:
            self._store.remove_document(kind, str(Path(path).resolve()))
            log.info("removed %s", path)
        except Exception as exc:  # noqa: BLE001 — same rule as _reindex
            log.warning("remove failed for %s: %s", path, exc)


class EventRelay(FileSystemEventHandler):
    """The handler the observer calls: it only queues each event, on watchdog's thread.

    :meth:`drain` then hands the queued events to a real handler on the *calling* thread — the
    one that owns the store.
    """

    def __init__(self) -> None:
        self._events: queue.SimpleQueue[FileSystemEvent] = queue.SimpleQueue()

    def dispatch(self, event: FileSystemEvent) -> None:
        self._events.put(event)

    def drain(self, handler: FileSystemEventHandler, timeout: float = 0.0) -> int:
        """Dispatch queued events to ``handler`` on this thread, in order; returns how many.

        Waits up to ``timeout`` seconds for the first event, then takes only what is already
        queued.
        """
        count = 0
        try:
            event = self._events.get(timeout=timeout) if timeout > 0 else self._events.get_nowait()
            while True:
                handler.dispatch(event)
                count += 1
                event = self._events.get_nowait()
        except queue.Empty:
            return count


def watch(paths: list[Path], store: Store, embedder: Embedder) -> None:
    """Watch ``paths`` recursively and keep the index in sync. Blocks until interrupted.

    Must be called on the thread that opened ``store``: index work runs here, not on the observer
    thread.
    """
    handler = CodeChangeHandler(store, embedder)
    relay = EventRelay()
    observer = Observer()
    for p in paths:
        observer.schedule(relay, str(Path(p).resolve()), recursive=True)
    observer.start()
    log.info("watching %s (Ctrl-C to stop)", ", ".join(str(p) for p in paths))
    try:
        while observer.is_alive():
            relay.drain(handler, timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
