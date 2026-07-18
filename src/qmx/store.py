"""SQLite store — ``sqlite-vec`` (vectors) + FTS5 (BM25) + hash tables.

The DB is a **rebuildable shadow** of the on-disk source (``plan/qmx-plan.md``): files are truth,
this is a cache.

Schema v3 separates chunk **content** from its **mentions** — the robustness core:

- ``chunks`` — one row per unique chunk text (``chunk_hash``), embedded **once**. Shared identical
  code across files dedups to a single embedding.
- ``mentions`` — where a chunk appears in a document (doc, ord, lines, symbol). Many mentions can
  point at one chunk; deleting a document drops its mentions (FK cascade).
- A chunk with **zero mentions is a tombstone**: excluded from search but kept so a rename/re-add
  reuses its warm embedding. :meth:`Store.purge_orphans` hard-deletes tombstones.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 4
_IN_BATCH = 500  # max params per IN(...) chunk


class StoreSchemaMismatch(RuntimeError):
    """The DB was built with a different embedding model/dim/schema — it must be rebuilt."""


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


@dataclass(slots=True)
class Learning:
    """One distilled lesson (Capability #3). ``superseded_by``/``promoted_to`` track lifecycle."""

    learning_id: int
    type: str
    topic: str | None
    scope: str | None
    statement: str
    detail: str | None
    importance: float
    source_anchors: str | None
    superseded_by: int | None
    reuse_count: int
    last_fired_at: str | None
    promoted_to: str | None
    created_at: str
    updated_at: str
    doc_id: int | None

    @property
    def is_live(self) -> bool:
        return self.superseded_by is None


@dataclass(slots=True)
class ReindexResult:
    """Outcome of reindexing one document."""

    embedded: int = 0  # new content chunks that required an embedding
    reused: int = 0  # distinct chunks already present (dedup / unchanged)
    mentions: int = 0  # mentions written for the document
    orphaned: int = 0  # chunks that lost their last mention (now tombstones)


class Store:
    """Owns the SQLite connection and the vector/FTS/mentions schema."""

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
        # Wait (not error) when another writer holds the lock — watch + capture + refresh can all
        # write the flat KB concurrently (WAL allows one writer at a time).
        conn.execute("PRAGMA busy_timeout=5000")
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
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._create_schema()
            version = SCHEMA_VERSION
        elif version > SCHEMA_VERSION:
            raise StoreSchemaMismatch(
                f"DB schema v{version} is newer than code v{SCHEMA_VERSION}; upgrade qmx"
            )
        while version < SCHEMA_VERSION:  # additive in-place upgrades (no rebuild needed)
            version = self._apply_migration(version)
        self._conn.execute(f"PRAGMA user_version={version}")
        self._conn.commit()
        self._check_embed_meta()

    def _apply_migration(self, version: int) -> int:
        """Apply the one incremental step from ``version`` and return the new version."""
        if version == 3:  # v3 -> v4: add the learnings tier (learnings + consolidated cursor)
            self._create_learnings_schema()
            return 4
        raise StoreSchemaMismatch(f"no migration path from schema v{version}")

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
                chunk_hash TEXT NOT NULL UNIQUE,   -- unique content -> embedded once
                text       TEXT NOT NULL
            );

            CREATE TABLE mentions (
                mention_id INTEGER PRIMARY KEY,
                doc_id     INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id),
                ord        INTEGER NOT NULL DEFAULT 0,
                start_line INTEGER,
                end_line   INTEGER,
                symbol     TEXT
            );
            CREATE INDEX idx_mentions_doc ON mentions(doc_id);
            CREATE INDEX idx_mentions_chunk ON mentions(chunk_id);
            """
        )
        c.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0("
            "chunk_id INTEGER PRIMARY KEY, "
            f"embedding float[{self._embed_dim}] distance_metric=cosine)"
        )
        c.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(text)")
        c.execute("INSERT INTO meta(key, value) VALUES('embed_dim', ?)", (str(self._embed_dim),))
        c.execute("INSERT INTO meta(key, value) VALUES('embed_model', ?)", (self._embed_model,))
        self._create_learnings_schema()

    def _create_learnings_schema(self) -> None:
        """The learnings tier (Capability #3): distilled lessons + the consolidation cursor.

        A learning's ``statement``+``detail`` is *also* embedded as a ``kind='learning'`` document
        (one chunk, ``path='learning:<id>'``) so retrieval reuses the vector+FTS pipeline unchanged;
        this table holds the structured metadata (type/scope/importance/supersede/promotion).
        """
        self._conn.executescript(
            """
            CREATE TABLE learnings (
                learning_id   INTEGER PRIMARY KEY,
                doc_id        INTEGER REFERENCES documents(doc_id) ON DELETE SET NULL,
                type          TEXT NOT NULL,         -- decision | mistake | howto
                topic         TEXT,
                scope         TEXT,                  -- canonical repo key, or NULL = global
                statement     TEXT NOT NULL,
                detail        TEXT,
                importance    REAL NOT NULL DEFAULT 0.5,   -- 0..1, used in retrieval ranking
                source_anchors TEXT,                 -- JSON citations
                superseded_by INTEGER REFERENCES learnings(learning_id),
                reuse_count   INTEGER NOT NULL DEFAULT 0,  -- promotion gate
                last_fired_at TEXT,
                promoted_to   TEXT,                  -- curated memory/*.md path it graduated to
                created_at    TEXT DEFAULT (datetime('now')),
                updated_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_learnings_scope ON learnings(scope);
            CREATE INDEX idx_learnings_doc ON learnings(doc_id);

            CREATE TABLE consolidated (   -- restart-safe cursor: chat chunks already distilled
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                at       TEXT DEFAULT (datetime('now'))
            );
            """
        )

    def _check_embed_meta(self) -> None:
        rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        stored_dim = int(rows.get("embed_dim", self._embed_dim))
        stored_model = rows.get("embed_model", self._embed_model)
        if stored_dim != self._embed_dim or stored_model != self._embed_model:
            raise StoreSchemaMismatch(
                f"index built with {stored_model!r}/dim {stored_dim}, "
                f"but config is {self._embed_model!r}/dim {self._embed_dim}; rebuild the index"
            )

    # -- documents -----------------------------------------------------------------------------

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

    def document_id(self, kind: str, path: str) -> int | None:
        """The ``doc_id`` for ``(kind, path)``, or ``None`` if not indexed."""
        row = self._conn.execute(
            "SELECT doc_id FROM documents WHERE kind=? AND path=?", (kind, path)
        ).fetchone()
        return None if row is None else row[0]

    def list_documents(self, kind: str) -> list[tuple[int, str]]:
        """``(doc_id, path)`` for every document of ``kind`` (e.g. all chat transcripts)."""
        rows = self._conn.execute(
            "SELECT doc_id, path FROM documents WHERE kind=? ORDER BY doc_id", (kind,)
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def documents_under(self, kind: str, path_prefix: str) -> list[tuple[int, str]]:
        """``(doc_id, path)`` for documents whose path starts with ``path_prefix``."""
        rows = self._conn.execute(
            "SELECT doc_id, path FROM documents WHERE kind=? AND path LIKE ? ESCAPE '\\'",
            (kind, _like_prefix(path_prefix)),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def remove_document(self, kind: str, path: str) -> int:
        """Delete a document by ``kind, path``; returns the number of chunks it orphaned."""
        row = self._conn.execute(
            "SELECT doc_id FROM documents WHERE kind=? AND path=?", (kind, path)
        ).fetchone()
        return 0 if row is None else self.remove_document_by_id(row[0])

    def remove_document_by_id(self, doc_id: int) -> int:
        """Delete a document (cascading its mentions); returns chunks left with no mentions."""
        candidate_ids = {
            r[0]
            for r in self._conn.execute(
                "SELECT chunk_id FROM mentions WHERE doc_id=?", (doc_id,)
            ).fetchall()
        }
        with self._conn:
            self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return self._count_orphans(candidate_ids)

    def remove_source(self, path: str, kind: str = "code") -> tuple[int, int]:
        """Remove the document at ``path`` and everything under ``path/``.

        Handles both a single file and a whole directory subtree. Returns
        ``(documents_removed, chunks_orphaned)``; run :meth:`purge_orphans` to reclaim space.
        """
        rows = self._conn.execute(
            "SELECT doc_id FROM documents WHERE kind=? AND (path=? OR path LIKE ? ESCAPE '\\')",
            (kind, path, _like_prefix(path.rstrip("/") + "/")),
        ).fetchall()
        docs = 0
        orphaned = 0
        for (doc_id,) in rows:
            orphaned += self.remove_document_by_id(doc_id)
            docs += 1
        return docs, orphaned

    def list_sources(self, kind: str | None = None) -> list[dict]:
        """Indexed sources grouped by ``repo``: document/chunk counts + a sample path."""
        where = "WHERE d.kind = ?" if kind is not None else ""
        params = (kind,) if kind is not None else ()
        rows = self._conn.execute(
            f"""
            SELECT d.repo AS repo,
                   count(DISTINCT d.doc_id) AS documents,
                   count(m.mention_id) AS chunks,
                   min(d.path) AS sample_path
            FROM documents d
            LEFT JOIN mentions m ON m.doc_id = d.doc_id
            {where}
            GROUP BY d.repo
            ORDER BY documents DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "repo": r["repo"],
                "documents": r["documents"],
                "chunks": r["chunks"],
                "sample_path": r["sample_path"],
            }
            for r in rows
        ]

    # -- indexing ------------------------------------------------------------------------------

    def missing_chunk_hashes(self, hashes: Iterable[str]) -> set[str]:
        """Which of ``hashes`` are not yet stored (i.e. still need embedding)."""
        wanted = set(hashes)
        if not wanted:
            return set()
        found: set[str] = set()
        items = list(wanted)
        for start in range(0, len(items), _IN_BATCH):
            batch = items[start : start + _IN_BATCH]
            placeholders = ",".join("?" * len(batch))
            found.update(
                r[0]
                for r in self._conn.execute(
                    f"SELECT chunk_hash FROM chunks WHERE chunk_hash IN ({placeholders})", batch
                ).fetchall()
            )
        return wanted - found

    def reindex_document(
        self,
        doc_id: int,
        chunks: Sequence[Chunk],
        new_embeddings: dict[str, Sequence[float]],
    ) -> ReindexResult:
        """Replace ``doc_id``'s mentions with ``chunks``, embedding only content not already stored.

        ``new_embeddings`` must supply a vector for every chunk hash returned by
        :meth:`missing_chunk_hashes` for these chunks. Existing content (dedup, unchanged, or a
        tombstone being revived) is reused without re-embedding.
        """
        result = ReindexResult()
        with self._conn:  # one transaction — crash-safe per document
            hash_to_id: dict[str, int] = {}
            for chunk in chunks:
                h = chunk.chunk_hash
                if h in hash_to_id:
                    continue
                row = self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE chunk_hash=?", (h,)
                ).fetchone()
                if row is not None:
                    hash_to_id[h] = row[0]
                    result.reused += 1
                else:
                    hash_to_id[h] = self._insert_content(h, chunk.text, new_embeddings)
                    result.embedded += 1

            old_ids = {
                r[0]
                for r in self._conn.execute(
                    "SELECT chunk_id FROM mentions WHERE doc_id=?", (doc_id,)
                ).fetchall()
            }
            self._conn.execute("DELETE FROM mentions WHERE doc_id=?", (doc_id,))
            for chunk in chunks:
                self._conn.execute(
                    """
                    INSERT INTO mentions(doc_id, chunk_id, ord, start_line, end_line, symbol)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        hash_to_id[chunk.chunk_hash],
                        chunk.ord,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.symbol,
                    ),
                )
            result.mentions = len(chunks)
            result.orphaned = self._count_orphans(old_ids - set(hash_to_id.values()))
        return result

    def _insert_content(
        self, chunk_hash: str, text: str, new_embeddings: dict[str, Sequence[float]]
    ) -> int:
        vector = new_embeddings.get(chunk_hash)
        if vector is None:
            raise ValueError(f"no embedding supplied for new chunk {chunk_hash[:12]}")
        if len(vector) != self._embed_dim:
            raise ValueError(f"embedding dim {len(vector)} != {self._embed_dim}")
        cur = self._conn.execute(
            "INSERT INTO chunks(chunk_hash, text) VALUES(?, ?)", (chunk_hash, text)
        )
        chunk_id = cur.lastrowid
        self._conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(list(vector))),
        )
        self._conn.execute("INSERT INTO fts_chunks(rowid, text) VALUES(?, ?)", (chunk_id, text))
        return chunk_id

    def _count_orphans(self, candidate_ids: set[int]) -> int:
        return sum(
            1
            for cid in candidate_ids
            if self._conn.execute(
                "SELECT 1 FROM mentions WHERE chunk_id=? LIMIT 1", (cid,)
            ).fetchone()
            is None
        )

    def purge_orphans(self) -> int:
        """Hard-delete tombstoned chunks (zero mentions) and their vectors/FTS rows."""
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT chunk_id FROM chunks c "
                "WHERE NOT EXISTS(SELECT 1 FROM mentions m WHERE m.chunk_id=c.chunk_id)"
            ).fetchall()
        ]
        if not ids:
            return 0
        with self._conn:
            for cid in ids:
                self._conn.execute("DELETE FROM vec_chunks WHERE chunk_id=?", (cid,))
                self._conn.execute("DELETE FROM fts_chunks WHERE rowid=?", (cid,))
                self._conn.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
        return len(ids)

    # -- reads ---------------------------------------------------------------------------------

    def get_chunk(self, chunk_id: int) -> SearchHit | None:
        """Fetch one live chunk's full text + representative location, or ``None`` if gone."""
        row = self._conn.execute(
            f"""
            SELECT c.chunk_id, 0.0 AS distance, c.text,
                   m.doc_id, m.start_line, m.end_line, m.symbol, d.kind, d.path
            FROM chunks c
            JOIN mentions m ON m.mention_id = {_REP_MENTION}
            JOIN documents d ON d.doc_id = m.doc_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return _rows_to_hits([row], 1, None, distance_key="distance")[0]

    def search_vec(
        self, query_embedding: Sequence[float], k: int = 10, kind: str | None = None
    ) -> list[SearchHit]:
        """Cosine top-k over live (mentioned) chunks, optionally filtered by ``kind``."""
        if len(query_embedding) != self._embed_dim:
            raise ValueError(f"query dim {len(query_embedding)} != {self._embed_dim}")
        # Over-fetch: the ANN table may still hold tombstoned (orphan) vectors that the mentions
        # join drops, and a kind filter trims further.
        fetch = max(k * 4, k + 20)
        rows = self._conn.execute(
            f"""
            SELECT v.chunk_id, v.distance, c.text,
                   m.doc_id, m.start_line, m.end_line, m.symbol, d.kind, d.path
            FROM vec_chunks v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN mentions m ON m.mention_id = {_REP_MENTION}
            JOIN documents d ON d.doc_id = m.doc_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(list(query_embedding)), fetch),
        ).fetchall()
        return _rows_to_hits(rows, k, kind, distance_key="distance")

    def search_fts(self, query: str, k: int = 10, kind: str | None = None) -> list[SearchHit]:
        """BM25 top-k over live chunks via FTS5, optionally filtered by ``kind``."""
        match = _fts_match_query(query)
        if match is None:
            return []
        fetch = max(k * 4, k + 20)
        rows = self._conn.execute(
            f"""
            SELECT f.rowid AS chunk_id, f.rank AS distance, c.text,
                   m.doc_id, m.start_line, m.end_line, m.symbol, d.kind, d.path
            FROM fts_chunks f
            JOIN chunks c ON c.chunk_id = f.rowid
            JOIN mentions m ON m.mention_id = {_REP_MENTION}
            JOIN documents d ON d.doc_id = m.doc_id
            WHERE fts_chunks MATCH ?
            ORDER BY f.rank
            LIMIT ?
            """,
            (match, fetch),
        ).fetchall()
        return _rows_to_hits(rows, k, kind, distance_key="distance")

    def counts(self) -> dict[str, int]:
        """Base row counts (documents, content chunks, vectors)."""
        q = lambda t: self._conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: E731
        return {"documents": q("documents"), "chunks": q("chunks"), "vectors": q("vec_chunks")}

    def index_stats(self) -> dict[str, int]:
        """Richer stats for ``status``: live vs tombstoned chunks and total mentions."""
        base = self.counts()
        orphans = self._conn.execute(
            "SELECT count(*) FROM chunks c "
            "WHERE NOT EXISTS(SELECT 1 FROM mentions m WHERE m.chunk_id=c.chunk_id)"
        ).fetchone()[0]
        base["mentions"] = self._conn.execute("SELECT count(*) FROM mentions").fetchone()[0]
        base["live_chunks"] = base["chunks"] - orphans
        base["tombstoned_chunks"] = orphans
        return base

    # -- learnings (kind="learning") -----------------------------------------------------------

    def insert_learning(
        self,
        *,
        type: str,
        statement: str,
        topic: str | None = None,
        scope: str | None = None,
        detail: str | None = None,
        importance: float = 0.5,
        source_anchors: str | None = None,
        doc_id: int | None = None,
    ) -> int:
        """Insert a learnings row; returns its ``learning_id``. Embedding is done separately."""
        cur = self._conn.execute(
            """
            INSERT INTO learnings(
                doc_id, type, topic, scope, statement, detail, importance, source_anchors)
            VALUES(:doc_id, :type, :topic, :scope, :statement, :detail, :importance, :anchors)
            RETURNING learning_id
            """,
            {
                "doc_id": doc_id,
                "type": type,
                "topic": topic,
                "scope": scope,
                "statement": statement,
                "detail": detail,
                "importance": importance,
                "anchors": source_anchors,
            },
        )
        learning_id = cur.fetchone()[0]
        self._conn.commit()
        return learning_id

    def set_learning_doc(self, learning_id: int, doc_id: int) -> None:
        """Link a learning to the ``kind='learning'`` document holding its embedded chunk."""
        with self._conn:
            self._conn.execute(
                "UPDATE learnings SET doc_id=?, updated_at=datetime('now') WHERE learning_id=?",
                (doc_id, learning_id),
            )

    def get_learning(self, learning_id: int) -> Learning | None:
        row = self._conn.execute(
            "SELECT * FROM learnings WHERE learning_id=?", (learning_id,)
        ).fetchone()
        return None if row is None else _row_to_learning(row)

    def learning_by_doc_id(self, doc_id: int) -> Learning | None:
        row = self._conn.execute("SELECT * FROM learnings WHERE doc_id=?", (doc_id,)).fetchone()
        return None if row is None else _row_to_learning(row)

    def list_learnings(
        self,
        *,
        scope: str | None = None,
        include_global: bool = False,
        live_only: bool = True,
        limit: int | None = None,
    ) -> list[Learning]:
        """Learnings ordered by importance then recency (the query-free injection ranking).

        ``scope`` filters to that repo key; ``include_global`` also pulls ``scope IS NULL`` rows.
        ``live_only`` excludes superseded lessons.
        """
        clauses: list[str] = []
        params: list[object] = []
        if scope is not None:
            if include_global:
                clauses.append("(scope = ? OR scope IS NULL)")
                params.append(scope)
            else:
                clauses.append("scope = ?")
                params.append(scope)
        elif not include_global:
            pass  # no scope filter -> all scopes
        if live_only:
            clauses.append("superseded_by IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM learnings {where} ORDER BY importance DESC, updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_row_to_learning(r) for r in self._conn.execute(sql, params).fetchall()]

    def supersede_learning(self, old_id: int, new_id: int) -> None:
        """Mark ``old_id`` as replaced by ``new_id`` (kept for audit, excluded from retrieval)."""
        with self._conn:
            self._conn.execute(
                "UPDATE learnings SET superseded_by=?, updated_at=datetime('now') "
                "WHERE learning_id=?",
                (new_id, old_id),
            )

    def update_learning(
        self,
        learning_id: int,
        *,
        statement: str | None = None,
        detail: str | None = None,
        importance: float | None = None,
        source_anchors: str | None = None,
    ) -> None:
        """Patch a live learning's content fields (used by consolidate's ``update`` decision)."""
        sets: list[str] = []
        params: list[object] = []
        for col, val in (
            ("statement", statement),
            ("detail", detail),
            ("importance", importance),
            ("source_anchors", source_anchors),
        ):
            if val is not None:
                sets.append(f"{col}=?")
                params.append(val)
        if not sets:
            return
        sets.append("updated_at=datetime('now')")
        params.append(learning_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE learnings SET {', '.join(sets)} WHERE learning_id=?", params
            )

    def touch_learning(self, learning_id: int) -> None:
        """Record a lesson firing (retrieved/injected): bump ``reuse_count`` + ``last_fired_at``."""
        with self._conn:
            self._conn.execute(
                "UPDATE learnings SET reuse_count = reuse_count + 1, "
                "last_fired_at = datetime('now') WHERE learning_id=?",
                (learning_id,),
            )

    def set_promoted(self, learning_id: int, path: str) -> None:
        """Record that a learning graduated to the curated memory file at ``path``."""
        with self._conn:
            self._conn.execute(
                "UPDATE learnings SET promoted_to=?, updated_at=datetime('now') "
                "WHERE learning_id=?",
                (path, learning_id),
            )

    # -- consolidation cursor ------------------------------------------------------------------

    def mark_consolidated(self, chunk_ids: Iterable[int]) -> None:
        """Record chat chunks as distilled so a re-run never re-processes them (idempotent)."""
        rows = [(cid,) for cid in chunk_ids]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO consolidated(chunk_id) VALUES(?)", rows
            )

    def unconsolidated_chat_chunks(self, doc_id: int) -> list[SearchHit]:
        """Live chat chunks of ``doc_id`` not yet in the consolidation cursor (transcript order)."""
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, 0.0 AS distance, c.text,
                   m.doc_id, m.start_line, m.end_line, m.symbol, d.kind, d.path
            FROM mentions m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            JOIN documents d ON d.doc_id = m.doc_id
            WHERE m.doc_id = ?
              AND c.chunk_id NOT IN (SELECT chunk_id FROM consolidated)
            ORDER BY m.ord
            """,
            (doc_id,),
        ).fetchall()
        return _rows_to_hits(rows, len(rows), None, distance_key="distance")


# One mention per chunk for display metadata (lowest mention_id). Correlated so orphan chunks —
# with no mention — yield NULL and are dropped by the INNER JOIN, excluding tombstones from search.
_REP_MENTION = "(SELECT MIN(mm.mention_id) FROM mentions mm WHERE mm.chunk_id = c.chunk_id)"

_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _rows_to_hits(
    rows: Sequence[sqlite3.Row], k: int, kind: str | None, *, distance_key: str
) -> list[SearchHit]:
    hits = [
        SearchHit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            kind=r["kind"],
            path=r["path"],
            text=r["text"],
            distance=r[distance_key],
            start_line=r["start_line"],
            end_line=r["end_line"],
            symbol=r["symbol"],
        )
        for r in rows
        if kind is None or r["kind"] == kind
    ]
    return hits[:k]


def _row_to_learning(r: sqlite3.Row) -> Learning:
    return Learning(
        learning_id=r["learning_id"],
        type=r["type"],
        topic=r["topic"],
        scope=r["scope"],
        statement=r["statement"],
        detail=r["detail"],
        importance=r["importance"],
        source_anchors=r["source_anchors"],
        superseded_by=r["superseded_by"],
        reuse_count=r["reuse_count"],
        last_fired_at=r["last_fired_at"],
        promoted_to=r["promoted_to"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        doc_id=r["doc_id"],
    )


def _like_prefix(prefix: str) -> str:
    """Escape LIKE wildcards in ``prefix`` and append ``%`` for a prefix match."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _fts_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression: OR of quoted alnum tokens."""
    tokens = _FTS_TOKEN.findall(text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
