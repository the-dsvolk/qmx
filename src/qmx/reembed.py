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
) -> ReembedStats:
    """Rebuild ``vec_chunks`` in ``db_path`` with ``embedder`` (recorded in ``meta`` as ``model``).

    ``progress(done, todo)`` is called after each committed batch.
    """
    conn = sqlite3.connect(str(db_path))
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

        for start in range(0, len(todo), commit_every):
            ids = todo[start : start + commit_every]
            marks = ",".join("?" * len(ids))
            text_by_id = dict(
                conn.execute(f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({marks})", ids)
            )
            vectors = embedder.embed([text_by_id[cid] for cid in ids])
            for vec in vectors:
                if len(vec) != dim:
                    raise ValueError(f"embedding dim {len(vec)} != {dim}")
            with conn:
                conn.executemany(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)",
                    [
                        (cid, sqlite_vec.serialize_float32(list(vec)))
                        for cid, vec in zip(ids, vectors, strict=True)
                    ],
                )
            stats.embedded += len(ids)
            if progress is not None:
                progress(stats.embedded, len(todo))
        return stats
    finally:
        conn.close()
