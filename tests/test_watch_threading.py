"""The watcher's index work must run on the thread that owns the store.

watchdog calls handlers on its observer thread, but a ``sqlite3`` connection refuses use from any
thread other than the one that opened it. These tests build the store on the test's thread (as
``qmx watch`` does on the main thread) and deliver events from another one.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileMovedEvent
from watchdog.observers.api import BaseObserver

from qmx.store import Store
from qmx.watch import CodeChangeHandler, EventRelay, watch
from tests.fakes import FakeEmbedder

FUNC = "def area(w, h):\n    return w * h\n"


def _doc_names(store: Store) -> set[str]:
    return {Path(p).name for _, p in store.list_documents("code")}


def _from_other_thread(fn, *args) -> None:
    """Run ``fn`` on a fresh thread (standing in for watchdog's observer) and re-raise its error."""
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fn(*args)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the test below
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if errors:
        raise errors[0]


def test_relay_applies_observer_thread_events_on_the_store_thread(tmp_path, caplog):
    embedder = FakeEmbedder(dim=64)
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as store:
        handler = CodeChangeHandler(store, embedder)
        relay = EventRelay()
        alpha, beta = tmp_path / "alpha.py", tmp_path / "beta.py"

        with caplog.at_level(logging.WARNING, logger="qmx.watch"):
            alpha.write_text(FUNC)
            _from_other_thread(relay.dispatch, FileCreatedEvent(str(alpha)))
            assert relay.drain(handler) == 1
            assert _doc_names(store) == {"alpha.py"}

            alpha.rename(beta)
            _from_other_thread(relay.dispatch, FileMovedEvent(str(alpha), str(beta)))
            assert relay.drain(handler) == 1
            assert _doc_names(store) == {"beta.py"}

            beta.unlink()
            _from_other_thread(relay.dispatch, FileDeletedEvent(str(beta)))
            assert relay.drain(handler) == 1
            assert _doc_names(store) == set()

        assert caplog.records == []  # no "reindex failed" / "remove failed"


def test_drain_waits_for_the_first_event_then_returns(tmp_path):
    embedder = FakeEmbedder(dim=64)
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as store:
        handler = CodeChangeHandler(store, embedder)
        relay = EventRelay()
        assert relay.drain(handler) == 0  # nothing queued: returns at once
        start = time.monotonic()
        assert relay.drain(handler, timeout=0.2) == 0
        assert time.monotonic() - start >= 0.15


def test_remove_failure_is_logged_not_raised(tmp_path, caplog):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    handler = CodeChangeHandler(store, embedder)
    store.close()  # any store error on delete — it must not kill the watcher
    with caplog.at_level(logging.WARNING, logger="qmx.watch"):
        handler.on_deleted(FileDeletedEvent(str(tmp_path / "gone.py")))
    assert "remove failed" in caplog.text


def _indexed_names(db: Path) -> set[str]:
    """Indexed code paths, read through a separate connection (as another process would)."""
    with closing(sqlite3.connect(db, timeout=5)) as conn:
        return {Path(p).name for (p,) in conn.execute("SELECT path FROM documents")}


def _wait_for(cond, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _stop_observers() -> None:
    if not _wait_for(lambda: any(isinstance(t, BaseObserver) for t in threading.enumerate()), 5):
        return
    for t in threading.enumerate():
        if isinstance(t, BaseObserver):
            t.stop()


def test_watch_indexes_real_file_events_with_a_store_from_the_calling_thread(tmp_path):
    """End to end: a real observer, real file edits, and the store opened on this thread."""
    root = tmp_path / "src"
    root.mkdir()
    db = tmp_path / "index.db"
    embedder = FakeEmbedder(dim=64)
    store = Store.open(db, embedder.dim, "fake")
    alpha, beta = root / "alpha.py", root / "beta.py"
    seen: dict[str, bool] = {}
    errors: list[BaseException] = []

    def touch_until_indexed() -> bool:
        deadline = time.monotonic() + 10
        n = 0
        while time.monotonic() < deadline:  # rewrite until seen: covers observer start-up
            n += 1
            alpha.write_text(f"def area(w, h):\n    return w * h * {n}\n")
            if _wait_for(lambda: "alpha.py" in _indexed_names(db), 0.5):
                return True
        return False

    def drive() -> None:
        try:
            seen["modify"] = touch_until_indexed()
            if seen["modify"]:
                alpha.rename(beta)
                seen["move"] = _wait_for(lambda: _indexed_names(db) == {"beta.py"})
                beta.unlink()
                seen["delete"] = _wait_for(lambda: _indexed_names(db) == set())
        except BaseException as exc:  # noqa: BLE001 — surfaced to the test below
            errors.append(exc)
        finally:
            _stop_observers()  # makes watch() return

    driver = threading.Thread(target=drive, daemon=True)
    driver.start()
    try:
        watch([root], store, embedder)  # blocks on this thread until the driver stops it
    finally:
        driver.join(timeout=60)
        store.close()

    assert not errors
    assert seen == {"modify": True, "move": True, "delete": True}
