"""Parked items — ideas set aside mid-chat to come back to later (``plan/qmx-parked.md``).

The markdown files under ``parked_root`` are the **source of truth**; nothing lives only in the DB:

- ``<root>/<NNNN>-<slug>.md`` — an **open** item; ``<root>/archive/`` holds ``done``/``dropped``
  ones. Closing moves a file, never deletes it, so ``reopen`` is just the move back.
- Each open item is also embedded as one ``kind='parked'`` document (``path='parked:<id>'``) so
  listing can rank by meaning and ``park`` can point at similar open items. That index is a
  **rebuildable shadow**: :func:`sync_index` reconciles it with the files, and every file operation
  here succeeds with the model backend down (embedding is best-effort, see :func:`try_index`).

Expiry is a soft **stale review** only: an open item untouched for ``stale_days`` is flagged
``stale`` for a keep-or-drop decision, never archived automatically.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qmx.embed import EmbedBackendError, Embedder
from qmx.index import reindex
from qmx.store import Chunk, Store, hash_text

log = logging.getLogger("qmx.parked")

KIND = "parked"
STATUSES = ("open", "done", "dropped")
CLOSED_STATUSES = ("done", "dropped")
DEFAULT_STALE_DAYS = 30

_PATH_PREFIX = "parked:"
_ARCHIVE = "archive"
_FILE_RE = re.compile(r"^(?P<id>\d+)-.*\.md$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_RESOLUTION = "## Resolution"
_FIELDS = ("id", "status", "scope", "created", "updated", "closed", "session")


@dataclass(frozen=True, slots=True)
class ParkedItem:
    """One parked idea, as read from (and written back to) its markdown file."""

    id: int
    title: str
    status: str = "open"
    scope: str | None = None  # canonical repo key, or None = global
    why: str | None = None
    note: str | None = None  # resolution, written on close
    created: str = ""
    updated: str = ""
    closed: str | None = None
    session: str | None = None  # transcript / session the idea came from, if known
    path: Path | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def is_stale(self, stale_days: int = DEFAULT_STALE_DAYS, now: datetime | None = None) -> bool:
        """Open and untouched for longer than ``stale_days`` — due a keep-or-drop review."""
        if not self.is_open or not self.updated:
            return False
        now = now or datetime.now(UTC)
        return now - _parse_ts(self.updated) > timedelta(days=stale_days)

    def embed_text(self) -> str:
        """Text indexed for this item (title + why; the trailing id keeps the chunk unique)."""
        body = self.title if not self.why else f"{self.title}\n\n{self.why}"
        return f"[parked] {body}\n[#{self.id}]"

    def to_dict(self, stale_days: int = DEFAULT_STALE_DAYS) -> dict:
        out = {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "scope": self.scope,
            "why": self.why,
            "created": self.created,
            "updated": self.updated,
            "stale": self.is_stale(stale_days),
            "path": str(self.path) if self.path else None,
        }
        if self.session:
            out["session"] = self.session
        if not self.is_open:
            out["closed"] = self.closed
            out["note"] = self.note
        return out


class ParkedNotFound(LookupError):
    """No parked item with that id (open or archived)."""


class ParkedRepo:
    """File-level operations on the parked directory — no DB, no model backend."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    @property
    def archive(self) -> Path:
        return self.root / _ARCHIVE

    # -- reads -------------------------------------------------------------------------------------

    def items(self, *, include_archived: bool = False) -> list[ParkedItem]:
        """All open items (plus archived ones if asked), oldest id first."""
        dirs = [self.root, self.archive] if include_archived else [self.root]
        out: list[ParkedItem] = []
        for d in dirs:
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_file() and _FILE_RE.match(p.name):
                    try:
                        out.append(read_item(p))
                    except (OSError, ValueError) as exc:
                        log.warning("skip unreadable parked file %s: %s", p, exc)
        return sorted(out, key=lambda i: i.id)

    def get(self, item_id: int) -> ParkedItem:
        for d in (self.root, self.archive):
            for p in d.iterdir() if d.is_dir() else ():
                m = _FILE_RE.match(p.name)
                if m and int(m["id"]) == item_id:
                    return read_item(p)
        raise ParkedNotFound(f"no parked item #{item_id}")

    # -- writes ------------------------------------------------------------------------------------

    def park(
        self,
        title: str,
        *,
        why: str | None = None,
        scope: str | None = None,
        session: str | None = None,
        now: datetime | None = None,
    ) -> ParkedItem:
        """Write a new open item; ids are monotonic and claimed with ``O_EXCL`` (race-safe)."""
        title = " ".join((title or "").split())
        if not title:
            raise ValueError("cannot park an empty idea")
        self.root.mkdir(parents=True, exist_ok=True)
        ts = _fmt_ts(now)
        item_id = self._next_id()
        while True:
            path = self.root / f"{item_id:04d}-{_slug(title)}.md"
            item = ParkedItem(
                id=item_id,
                title=title,
                scope=scope or None,
                why=(why or "").strip() or None,
                created=ts,
                updated=ts,
                session=session or None,
                path=path,
            )
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:  # a concurrent park took this id
                item_id = self._next_id(at_least=item_id + 1)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(render_item(item))
            return item

    def close(
        self,
        item_id: int,
        *,
        status: str = "done",
        note: str | None = None,
        now: datetime | None = None,
    ) -> ParkedItem:
        """Mark ``done``/``dropped`` and move the file into ``archive/`` (never deleted)."""
        if status not in CLOSED_STATUSES:
            raise ValueError(f"close status must be one of {CLOSED_STATUSES}, got {status!r}")
        item = self.get(item_id)
        ts = _fmt_ts(now)
        closed = replace(
            item,
            status=status,
            note=(note or "").strip() or item.note,
            updated=ts,
            closed=ts,
        )
        return self._move(closed, self.archive)

    def reopen(self, item_id: int, *, now: datetime | None = None) -> ParkedItem:
        """Undo a close: back to ``open`` in the live directory (the resolution note is kept)."""
        item = self.get(item_id)
        reopened = replace(item, status="open", closed=None, updated=_fmt_ts(now))
        return self._move(reopened, self.root)

    def keep(self, item_id: int, *, now: datetime | None = None) -> ParkedItem:
        """Stale review "still relevant": bump ``updated`` so it stops being flagged."""
        item = self.get(item_id)
        if not item.is_open:
            raise ValueError(f"parked item #{item_id} is {item.status}; reopen it first")
        return self._move(replace(item, updated=_fmt_ts(now)), self.root)

    def _move(self, item: ParkedItem, dest_dir: Path) -> ParkedItem:
        assert item.path is not None
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item.path.name
        moved = replace(item, path=dest)
        _atomic_write(dest, render_item(moved))
        if dest != item.path:
            item.path.unlink(missing_ok=True)
        return moved

    def _next_id(self, at_least: int = 1) -> int:
        ids = [
            int(m["id"])
            for d in (self.root, self.archive)
            if d.is_dir()
            for p in d.iterdir()
            if (m := _FILE_RE.match(p.name))
        ]
        return max([at_least - 1, *ids]) + 1


# -- shadow index ----------------------------------------------------------------------------------


def doc_path(item_id: int) -> str:
    """Synthetic document path for an item's embedded chunk (``parked:<id>``)."""
    return f"{_PATH_PREFIX}{item_id}"


def item_id_from_path(path: str | None) -> int | None:
    if not path or not path.startswith(_PATH_PREFIX):
        return None
    try:
        return int(path[len(_PATH_PREFIX) :])
    except ValueError:
        return None


def index_item(store: Store, embedder: Embedder, item: ParkedItem) -> bool:
    """Embed one open item (no-op if its text is unchanged). Raises if the backend is down."""
    text = item.embed_text()
    file_hash = hash_text(f"{text}\n{item.scope or ''}")
    path = doc_path(item.id)
    if store.document_hash(KIND, path) == file_hash:
        return False
    doc_id = store.upsert_document(KIND, path, repo=item.scope or "_global")
    reindex(store, embedder, doc_id, [Chunk(text=text, symbol="parked")])
    store.set_document_hash(doc_id, file_hash)
    return True


def try_index(store: Store, embedder: Embedder, item: ParkedItem) -> bool:
    """Best-effort :func:`index_item`: the file is already safe, so a down backend is not an error.

    Returns whether the item is now searchable; ``qmx refresh`` picks up anything left behind.
    """
    try:
        index_item(store, embedder, item)
        return True
    except EmbedBackendError as exc:
        log.info("parked #%s saved but not indexed (backend down): %s", item.id, exc)
        return False


def unindex_item(store: Store, item_id: int) -> None:
    store.remove_document(KIND, doc_path(item_id))


@dataclass(slots=True)
class SyncStats:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0


def sync_index(store: Store, embedder: Embedder, repo: ParkedRepo) -> SyncStats:
    """Reconcile the shadow index with the files: open items in, everything else out."""
    stats = SyncStats()
    open_items = repo.items()
    live = {i.id for i in open_items}
    for item in open_items:
        if index_item(store, embedder, item):
            stats.indexed += 1
        else:
            stats.unchanged += 1
    for _doc_id, path in store.list_documents(KIND):
        item_id = item_id_from_path(path)
        if item_id not in live:
            store.remove_document(KIND, path)
            stats.removed += 1
    return stats


def filter_scope(
    items: list[ParkedItem], scope: str | None, *, all_scopes: bool = False
) -> list[ParkedItem]:
    """Items for ``scope`` plus global ones; everything when ``all_scopes``."""
    if all_scopes:
        return items
    return [i for i in items if i.scope is None or i.scope == scope]


# -- file format -----------------------------------------------------------------------------------


def render_item(item: ParkedItem) -> str:
    meta = {
        "id": str(item.id),
        "status": item.status,
        "scope": item.scope or "",
        "created": item.created,
        "updated": item.updated,
        "closed": item.closed or "",
        "session": item.session or "",
    }
    lines = ["---", *(f"{k}: {v}" for k, v in meta.items() if v or k == "scope"), "---"]
    lines += [f"# {item.title}", ""]
    if item.why:
        lines += [item.why, ""]
    if item.note:
        lines += [_RESOLUTION, "", item.note, ""]
    return "\n".join(lines)


def read_item(path: Path) -> ParkedItem:
    """Parse an item file (tolerant of hand edits: unknown keys ignored, body free-form)."""
    text = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(text)
    if "id" not in meta:
        raise ValueError("missing 'id' in frontmatter")
    title, why, note = _split_body(body)
    status = meta.get("status") or "open"
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    return ParkedItem(
        id=int(meta["id"]),
        title=title or path.stem,
        status=status,
        scope=meta.get("scope") or None,
        why=why,
        note=note,
        created=meta.get("created", ""),
        updated=meta.get("updated") or meta.get("created", ""),
        closed=meta.get("closed") or None,
        session=meta.get("session") or None,
        path=path,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter")
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :])
        key, sep, value = line.partition(":")
        if sep and key.strip() in _FIELDS:
            meta[key.strip()] = value.strip()
    raise ValueError("unterminated frontmatter")


def _split_body(body: str) -> tuple[str, str | None, str | None]:
    head, sep, note = body.partition(_RESOLUTION)
    lines = head.strip().splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines.pop(0)[2:].strip()
    why = "\n".join(lines).strip() or None
    return title, why, (note.strip() or None) if sep else None


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _slug(title: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "item"


def _fmt_ts(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
