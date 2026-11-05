"""Re-embed an index with a different embedding model — only the vector table changes.

Everything else in the index (chunk text, keyword index, mentions, lessons and their history,
parked items) is model-independent, and every chunk's text is stored in ``chunks.text``. So a model
change never re-reads source files — which matters because some sources are gone (Claude Code
prunes old transcripts) — and never re-runs the judge. It rebuilds ``vec_chunks`` from
``chunks.text`` with the configured embedder's *document* side (``embed_passage_prefix``).

Tombstoned chunks (zero mentions) are re-embedded too: a revived tombstone reuses its stored vector
instead of being embedded again, so it must have one in the new space.

Resumable: the first run drops ``vec_chunks``, recreates it at the new width and records the new
model in ``meta``; a rerun on a DB whose ``meta`` already names the new model and width only embeds
chunks that have no vector yet. Run it on a *copy* (``sqlite3 index.db ".backup copy.db"``) and
switch ``db_path`` once the copy is checked — the store refuses to open a DB whose ``meta`` does not
match the configured model, so the live index cannot be half-converted by accident.

``reuse_from`` names an index already converted to the same model and width (e.g. the copy built
and checked earlier): vectors for chunks with the same content hash are copied from it instead of
re-embedded. That makes the cutover catch-up — a fresh copy of the live index taken with the
services stopped — cost only the chunks added since the first copy.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from qmx.embed import Embedder
from qmx.store import SCHEMA_VERSION, StoreSchemaMismatch, create_vec_table, load_sqlite_vec

DEFAULT_COMMIT_EVERY = 512


@dataclass
class ReembedStats:
    chunks_total: int = 0
    already_embedded: int = 0
    embedded: int = 0
    reused: int = 0
    rebuilt_table: bool = False
    old_model: str = ""
    old_dim: int = 0


def reembed(
    db_path: Path | str,
    embedder: Embedder,
    model: str,
    *,
    commit_every: int = DEFAULT_COMMIT_EVERY,
    progress: Callable[[int, int], None] | None = None,
    reuse_from: Path | str | None = None,
) -> ReembedStats:
    """Rebuild ``vec_chunks`` in ``db_path`` with ``embedder`` (recorded in ``meta`` as ``model``).

    ``progress(done, todo)`` is called after each committed batch. ``reuse_from``: see module doc.
    """
    conn = sqlite3.connect(str(db_path))
    reuse = _open_reuse(reuse_from, model, embedder.dim) if reuse_from is not None else None
    try:
        load_sqlite_vec(conn)
        conn.execute("PRAGMA busy_timeout=5000")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise StoreSchemaMismatch(
                f"{db_path}: schema v{version}, this qmx expects v{SCHEMA_VERSION}; "
                "open it once with a matching qmx first"
            )
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        stats = ReembedStats(
            old_model=meta.get("embed_model", ""), old_dim=int(meta.get("embed_dim", 0) or 0)
        )
        dim = embedder.dim
        if stats.old_model != model or stats.old_dim != dim:
            with conn:
                conn.execute("DROP TABLE IF EXISTS vec_chunks")
                create_vec_table(conn, dim)
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('embed_model', ?)", (model,)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('embed_dim', ?)", (str(dim),)
                )
            stats.rebuilt_table = True

        done_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM vec_chunks")}
        all_ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")]
        todo = [cid for cid in all_ids if cid not in done_ids]
        stats.chunks_total = len(all_ids)
        stats.already_embedded = len(all_ids) - len(todo)

        done = 0
        for start in range(0, len(todo), commit_every):
            ids = todo[start : start + commit_every]
            marks = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT chunk_id, chunk_hash, text FROM chunks WHERE chunk_id IN ({marks})", ids
            ).fetchall()
            blobs: dict[int, bytes] = {}
            if reuse is not None:
                by_hash = _reuse_blobs(reuse, [h for _, h, _ in rows])
                blobs = {cid: by_hash[h] for cid, h, _ in rows if h in by_hash}
            fresh = [(cid, text) for cid, _, text in rows if cid not in blobs]
            vectors = embedder.embed([text for _, text in fresh])
            for (cid, _), vec in zip(fresh, vectors, strict=True):
                if len(vec) != dim:
                    raise ValueError(f"embedding dim {len(vec)} != {dim}")
                blobs[cid] = sqlite_vec.serialize_float32(list(vec))
            with conn:
                conn.executemany(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)", blobs.items()
                )
            stats.embedded += len(fresh)
            stats.reused += len(rows) - len(fresh)
            done += len(rows)
            if progress is not None:
                progress(done, len(todo))
        return stats
    finally:
        conn.close()
        if reuse is not None:
            reuse.close()


def _open_reuse(path: Path | str, model: str, dim: int) -> sqlite3.Connection:
    """Read-only handle on an index already converted to ``model``/``dim``."""
    if not Path(path).is_file():
        raise ValueError(f"reuse index not found: {path}")
    rc = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    load_sqlite_vec(rc)
    meta = dict(rc.execute("SELECT key, value FROM meta").fetchall())
    if meta.get("embed_model") != model or int(meta.get("embed_dim", 0) or 0) != dim:
        rc.close()
        raise ValueError(
            f"reuse index {path} holds {meta.get('embed_model')!r}/dim {meta.get('embed_dim')}, "
            f"not {model!r}/dim {dim}"
        )
    return rc


def _reuse_blobs(rc: sqlite3.Connection, hashes: list[str]) -> dict[str, bytes]:
    """``chunk_hash -> stored vector blob`` for the hashes the reuse index has a vector for."""
    if not hashes:
        return {}
    marks = ",".join("?" * len(hashes))
    return dict(
        rc.execute(
            f"SELECT c.chunk_hash, v.embedding FROM chunks c "
            f"JOIN vec_chunks v ON v.chunk_id = c.chunk_id WHERE c.chunk_hash IN ({marks})",
            hashes,
        )
    )
