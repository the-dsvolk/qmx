"""Watcher routing: synthetic watchdog events drive incremental index / remove."""

from __future__ import annotations

import pytest
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from qmx.store import Store
from qmx.watch import CodeChangeHandler
from tests.fakes import FakeEmbedder

FUNC = "def f(x):\n    return x + 1\n"
FUNC2 = "def f(x):\n    return x + 2\n"


@pytest.fixture
def env(tmp_path):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    handler = CodeChangeHandler(store, embedder)
    yield tmp_path, store, handler
    store.close()


def test_create_and_modify_indexes_file(env):
    root, store, handler = env
    f = root / "a.py"
    f.write_text(FUNC)
    handler.on_created(FileCreatedEvent(str(f)))
    assert store.index_stats()["live_chunks"] == 1

    f.write_text(FUNC2)
    handler.on_modified(FileModifiedEvent(str(f)))
    # still one live chunk (edited), old one tombstoned
    assert store.index_stats()["live_chunks"] == 1
    assert store.index_stats()["tombstoned_chunks"] == 1


def test_delete_removes_document(env):
    root, store, handler = env
    f = root / "a.py"
    f.write_text(FUNC)
    handler.on_created(FileCreatedEvent(str(f)))
    assert store.counts()["documents"] == 1

    handler.on_deleted(FileDeletedEvent(str(f)))
    assert store.counts()["documents"] == 0
    assert store.index_stats()["live_chunks"] == 0


def test_non_code_files_are_ignored(env):
    root, store, handler = env
    txt = root / "notes.txt"
    txt.write_text("just prose, not code")
    handler.on_created(FileCreatedEvent(str(txt)))
    handler.on_modified(FileModifiedEvent(str(txt)))
    assert store.counts()["documents"] == 0


def test_directory_events_are_ignored(env):
    root, store, handler = env
    evt = FileModifiedEvent(str(root))
    evt.is_directory = True
    handler.on_modified(evt)
    assert store.counts()["documents"] == 0


def test_move_removes_source_and_indexes_dest(env):
    root, store, handler = env
    src = root / "a.py"
    src.write_text(FUNC)
    handler.on_created(FileCreatedEvent(str(src)))

    dest = root / "b.py"
    src.rename(dest)
    handler.on_moved(FileMovedEvent(str(src), str(dest)))

    paths = {p for _, p in store.documents_under("code", str(root) + "/")}
    assert any(p.endswith("b.py") for p in paths)
    assert not any(p.endswith("a.py") for p in paths)
