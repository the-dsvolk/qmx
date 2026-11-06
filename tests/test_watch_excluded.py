"""The watcher indexes only what a directory index of its root would, under the root's repo label.

Files under ``EXCLUDE_DIRS`` or a dot-folder (``.venv/``, ``build/``, ``node_modules/``,
``.claude/`` …) are pruned by :func:`qmx.index.iter_source_files`; the watcher must skip the same
paths. Otherwise a ``uv sync`` or a build in a watched repo floods the embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from qmx.index import index_paths, is_excluded, iter_source_files, repo_kind
from qmx.store import Store
from qmx.watch import CodeChangeHandler
from tests.fakes import FakeEmbedder

FUNC = "def area(w, h):\n    return w * h\n"

EXCLUDED = [
    ".venv/lib/python3.12/site-packages/fancylib/core.py",
    "build/generated/schema_pb2.py",
    "node_modules/left-pad/index.js",
    ".claude/worktrees/feature-x/notes.md",
]


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts embed calls."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim)
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed(texts)


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "demo_repo"
    root.mkdir()
    embedder = CountingEmbedder()
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    handler = CodeChangeHandler(store, embedder, [root])
    yield root, store, embedder, handler
    store.close()


def _write(path: Path, text: str = FUNC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _doc_names(store: Store) -> set[str]:
    return {Path(p).name for kind in ("code", "doc") for _, p in store.list_documents(kind)}


@pytest.mark.parametrize("rel", EXCLUDED)
def test_events_under_excluded_folders_are_ignored(env, rel):
    root, store, embedder, handler = env
    f = _write(root / rel)
    moved = f.with_name("renamed" + f.suffix)
    writes_before = store._conn.total_changes

    handler.on_created(FileCreatedEvent(str(f)))
    handler.on_modified(FileModifiedEvent(str(f)))
    f.rename(moved)
    handler.on_moved(FileMovedEvent(str(f), str(moved)))
    moved.unlink()
    handler.on_deleted(FileDeletedEvent(str(moved)))

    assert store._conn.total_changes == writes_before  # no store writes
    assert embedder.calls == 0
    assert _doc_names(store) == set()


def test_normal_files_next_to_excluded_folders_are_indexed(env):
    root, store, embedder, handler = env
    for rel in EXCLUDED:
        handler.on_created(FileCreatedEvent(str(_write(root / rel))))
    app = _write(root / "app.py")
    lib = _write(root / "src" / "geometry.py", "def perimeter(w, h):\n    return 2 * (w + h)\n")
    readme = _write(root / "README.md", "# Demo\n\nA made-up repo.\n")
    for f in (app, lib, readme):
        handler.on_created(FileCreatedEvent(str(f)))

    assert _doc_names(store) == {"app.py", "geometry.py", "README.md"}
    assert embedder.calls > 0


def test_moving_a_file_into_an_excluded_folder_drops_it(env):
    root, store, _, handler = env
    src = _write(root / "app.py")
    handler.on_created(FileCreatedEvent(str(src)))
    dest = root / "build" / "app.py"
    dest.parent.mkdir()
    src.rename(dest)
    handler.on_moved(FileMovedEvent(str(src), str(dest)))
    assert _doc_names(store) == set()


def test_an_indexed_file_in_an_excluded_folder_is_still_removed(env):
    """Delete / move-away of a file indexed some other way (e.g. ``qmx index <file>``)."""
    root, store, _, handler = env
    gone = _write(root / "build" / "generated" / "gone.py")
    kept = _write(root / "build" / "generated" / "kept.py", FUNC.replace("area", "volume"))
    index_paths([gone], store, FakeEmbedder())
    index_paths([kept], store, FakeEmbedder())
    assert _doc_names(store) == {"gone.py", "kept.py"}

    gone.unlink()
    handler.on_deleted(FileDeletedEvent(str(gone)))
    assert _doc_names(store) == {"kept.py"}

    dest = root / "kept.py"
    kept.rename(dest)  # moved out of the excluded folder: removed there, indexed here
    handler.on_moved(FileMovedEvent(str(kept), str(dest)))
    assert {p for _, p in store.list_documents("code")} == {str(dest.resolve())}


def test_a_root_inside_a_dot_folder_is_still_watched(tmp_path):
    """Exclusion is relative to the watched root, as in the directory walk."""
    root = tmp_path / ".claude" / "worktrees" / "demo_repo"
    embedder = FakeEmbedder()
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as store:
        handler = CodeChangeHandler(store, embedder, [root])
        f = _write(root / "app.py")
        handler.on_created(FileCreatedEvent(str(f)))
        assert _doc_names(store) == {"app.py"}


def test_watcher_records_the_watched_roots_repo_label(env, tmp_path):
    root, store, _, handler = env
    f = _write(root / "src" / "pkg" / "shapes.py")
    handler.on_created(FileCreatedEvent(str(f)))
    assert [s["repo"] for s in store.list_sources("code")] == ["demo_repo"]

    # Same label a directory index of the root records, so the two never fight over the row.
    with Store.open(tmp_path / "walk.db", 64, "fake") as walked:
        index_paths([root], walked, FakeEmbedder())
        assert [s["repo"] for s in walked.list_sources("code")] == ["demo_repo"]


def test_is_excluded_agrees_with_the_directory_walk(tmp_path):
    root = tmp_path / "demo_repo"
    rels = [*EXCLUDED, "app.py", "src/geometry.py", "docs/guide.md", ".eslintrc.js",
            "src/build/gen.py", "pkg/__pycache__/mod.py", "src/.cache/tmp.py"]  # fmt: skip
    files = [_write(root / rel) for rel in rels]
    walked = set(iter_source_files(root))
    predicted = {f for f in files if repo_kind(f) is not None and not is_excluded(f, root)}
    assert walked == predicted
    assert {f.name for f in walked} == {"app.py", "geometry.py", "guide.md", ".eslintrc.js"}
