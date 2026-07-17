"""SQLite store — ``sqlite-vec`` (vectors) + FTS5 (BM25) + hash tables.

The DB is a **rebuildable shadow** of the on-disk source (``plan/qmx-plan.md``): files are truth,
this is a cache. Phase 0 delivers the schema, migrations, chunk upsert with content-hash dedup, and
cosine top-k vector search — enough for the round-trip acceptance test.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 2


class StoreSchemaMismatch(RuntimeError):
    """The DB was built with a different embedding model/dim — it must be rebuilt."""


def hash_text(text: str) -> str:
    """Stable content hash used for per-chunk dedup and incremental reindex."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Chunk:
    """One indexable unit of a document."""

    text: str
    ord: int = 0
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None

    @property
    def chunk_hash(self) -> str:
        return hash_text(self.text)


@dataclass(slots=True)
class SearchHit:
    chunk_id: int
    doc_id: int
    kind: str
    path: str | None
    text: str
    distance: float
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None


class Store:
    """Owns the SQLite connection and the vector/FTS schema."""

    def __init__(self, conn: sqlite3.Connection, embed_dim: int, embed_model: str) -> None:
        self._conn = conn
        self._embed_dim = embed_dim
        self._embed_model = embed_model

    # -- lifecycle -----------------------------------------------------------------------------

    @classmethod
    def open(cls, db_path: Path | str, embed_dim: int, embed_model: str) -> Store:
        """Open (creating if needed) the index at ``db_path`` and run migrations."""
        path = Path(db_path)
        if path.parent and str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        _load_sqlite_vec(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        store = cls(conn, embed_dim, embed_model)
        store._migrate()
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schema --------------------------------------------------------------------------------

    def _migrate(self) -> None:
        cur = self._conn.execute("PRAGMA user_version")
        version = cur.fetchone()[0]
        if version == 0:
            self._create_schema()
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()
        elif version != SCHEMA_VERSION:
            raise StoreSchemaMismatch(
                f"DB schema v{version} != code v{SCHEMA_VERSION}; rebuild the index"
            )
        self._check_embed_meta()

    def _create_schema(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

            CREATE TABLE documents (
                doc_id     INTEGER PRIMARY KEY,
                kind       TEXT NOT NULL,          -- code | doc | chat | learning
                repo       TEXT,
                path       TEXT,
                mtime      REAL,
                file_hash  TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE UNIQUE INDEX idx_documents_kind_path ON documents(kind, path);

            CREATE TABLE chunks (
                chunk_id   INTEGER PRIMARY KEY,
                doc_id     INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                ord        INTEGER NOT NULL DEFAULT 0,
                text       TEXT NOT NULL,
                chunk_hash TEXT NOT NULL UNIQUE,   -- dedup: identical text embeds once
                start_line INTEGER,
                end_line   INTEGER,
                symbol     TEXT,
                tombstoned INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX idx_chunks_doc ON chunks(doc_id);
            """
        )
        # Vector table — dimension is fixed at creation; cosine distance to match search semantics.
        c.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0("
            "chunk_id INTEGER PRIMARY KEY, "
            f"embedding float[{self._embed_dim}] distance_metric=cosine)"
        )
        # FTS5 over chunk text for BM25. Plain (not contentless) so rows can be DELETEd on
        # re-index; external-content + triggers is a Phase 2 space optimization.
        c.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(text)")
        c.execute("INSERT INTO meta(key, value) VALUES('embed_dim', ?)", (str(self._embed_dim),))
        c.execute("INSERT INTO meta(key, value) VALUES('embed_model', ?)", (self._embed_model,))

    def _check_embed_meta(self) -> None:
        rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        stored_dim = int(rows.get("embed_dim", self._embed_dim))
        stored_model = rows.get("embed_model", self._embed_model)
        if stored_dim != self._embed_dim or stored_model != self._embed_model:
            raise StoreSchemaMismatch(
                f"index built with {stored_model!r}/dim {stored_dim}, "
                f"but config is {self._embed_model!r}/dim {self._embed_dim}; rebuild the index"
            )

    # -- writes --------------------------------------------------------------------------------

    def upsert_document(
        self,
        kind: str,
        path: str | None = None,
        *,
        repo: str | None = None,
        mtime: float | None = None,
        file_hash: str | None = None,
    ) -> int:
        """Insert or update a document row (unique on ``kind, path``); returns its ``doc_id``."""
        cur = self._conn.execute(
            """
            INSERT INTO documents(kind, repo, path, mtime, file_hash)
            VALUES(:kind, :repo, :path, :mtime, :file_hash)
            ON CONFLICT(kind, path) DO UPDATE SET
                repo=excluded.repo, mtime=excluded.mtime, file_hash=excluded.file_hash
            RETURNING doc_id
            """,
            {"kind": kind, "repo": repo, "path": path, "mtime": mtime, "file_hash": file_hash},
        )
        doc_id = cur.fetchone()[0]
        self._conn.commit()
        return doc_id

    def document_hash(self, kind: str, path: str) -> str | None:
        """The stored ``file_hash`` for a document, or ``None`` if not indexed yet."""
        row = self._conn.execute(
            "SELECT file_hash FROM documents WHERE kind=? AND path=?", (kind, path)
        ).fetchone()
        return row[0] if row is not None else None

    def add_chunks(
        self, doc_id: int, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> list[int]:
        """Store chunks + their embeddings under ``doc_id``. Dedups on ``chunk_hash``.

        Returns the ``chunk_id`` for each input chunk (existing id when a hash already present).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        ids: list[int] = []
        with self._conn:  # single transaction — crash-safe per batch
            for chunk, vector in zip(chunks, embeddings, strict=True):
                if len(vector) != self._embed_dim:
                    raise ValueError(f"embedding dim {len(vector)} != {self._embed_dim}")
                existing = self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE chunk_hash=?", (chunk.chunk_hash,)
                ).fetchone()
                if existing is not None:
                    ids.append(existing[0])
                    continue
                cur = self._conn.execute(
                    """
                    INSERT INTO chunks(doc_id, ord, text, chunk_hash, start_line, end_line, symbol)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk.ord,
                        chunk.text,
                        chunk.chunk_hash,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.symbol,
                    ),
                )
                chunk_id = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(list(vector))),
                )
                self._conn.execute(
                    "INSERT INTO fts_chunks(rowid, text) VALUES(?, ?)", (chunk_id, chunk.text)
                )
                ids.append(chunk_id)
        return ids

    def clear_document_chunks(self, doc_id: int) -> int:
        """Remove all chunks (and their vec/fts rows) for a document; returns the count removed.

        Used to re-index a changed file wholesale. NOTE: with the Phase 0 global-unique dedup a
        chunk lives under a single ``doc_id``; a chunk whose identical text also appears in another
        file is dropped here. The proper chunk<->document link table is Phase 2 robustness work.
        """
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchall()
        ]
        if not ids:
            return 0
        with self._conn:
            for cid in ids:
                self._conn.execute("DELETE FROM vec_chunks WHERE chunk_id=?", (cid,))
                self._conn.execute("DELETE FROM fts_chunks WHERE rowid=?", (cid,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return len(ids)

    # -- reads ---------------------------------------------------------------------------------

    def search_vec(
        self, query_embedding: Sequence[float], k: int = 10, kind: str | None = None
    ) -> list[SearchHit]:
        """Cosine top-k over live (non-tombstoned) chunks, optionally filtered by ``kind``."""
        if len(query_embedding) != self._embed_dim:
            raise ValueError(f"query dim {len(query_embedding)} != {self._embed_dim}")
        # Over-fetch when filtering so the post-filter still yields up to k live hits.
        fetch = k * 5 if kind is not None else k
        rows = self._conn.execute(
            """
            SELECT v.chunk_id, v.distance, c.doc_id, c.text, c.start_line, c.end_line, c.symbol,
                   d.kind, d.path
            FROM vec_chunks v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE v.embedding MATCH ? AND k = ? AND c.tombstoned = 0
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(list(query_embedding)), fetch),
        ).fetchall()
        hits = [
            SearchHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                kind=r["kind"],
                path=r["path"],
                text=r["text"],
                distance=r["distance"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                symbol=r["symbol"],
            )
            for r in rows
            if kind is None or r["kind"] == kind
        ]
        return hits[:k]

    def search_fts(self, query: str, k: int = 10, kind: str | None = None) -> list[SearchHit]:
        """BM25 top-k over live chunks via FTS5, optionally filtered by ``kind``.

        ``distance`` carries the FTS5 ``rank`` (more negative = better); callers that fuse rankings
        (see ``search.py``) use position, not the raw value.
        """
        match = _fts_match_query(query)
        if match is None:
            return []
        fetch = k * 5 if kind is not None else k
        rows = self._conn.execute(
            """
            SELECT f.rowid AS chunk_id, f.rank AS rank, c.doc_id, c.text,
                   c.start_line, c.end_line, c.symbol, d.kind, d.path
            FROM fts_chunks f
            JOIN chunks c ON c.chunk_id = f.rowid
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE fts_chunks MATCH ? AND c.tombstoned = 0
            ORDER BY f.rank
            LIMIT ?
            """,
            (match, fetch),
        ).fetchall()
        hits = [
            SearchHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                kind=r["kind"],
                path=r["path"],
                text=r["text"],
                distance=r["rank"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                symbol=r["symbol"],
            )
            for r in rows
            if kind is None or r["kind"] == kind
        ]
        return hits[:k]

    def counts(self) -> dict[str, int]:
        """Row counts for ``status`` / smoke tests."""
        q = lambda t: self._conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: E731
        return {
            "documents": q("documents"),
            "chunks": q("chunks"),
            "vectors": q("vec_chunks"),
        }


_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _fts_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression: OR of quoted alnum tokens.

    Quoting each token neutralises FTS5 operators/syntax in user input; ``None`` when there is
    nothing to match.
    """
    tokens = _FTS_TOKEN.findall(text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
