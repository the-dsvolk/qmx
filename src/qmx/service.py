"""Service layer — the read API the MCP server (and tests) call.

One place that owns "open the store, run a search, shape a JSON-friendly result", so the MCP tools
stay thin. Each call opens a short-lived store connection (SQLite WAL handles concurrent readers);
the embedder/HTTP client is shared for the service's lifetime.
"""

from __future__ import annotations

import httpx

from qmx.config import Settings
from qmx.embed import Embedder, OllamaEmbedder
from qmx.search import search
from qmx.store import SearchHit, Store

MAX_TEXT_CHARS = 4000  # cap chunk text returned to an agent so results stay compact


class QmxService:
    """Read-side operations over the index: ``query``, ``get``, ``status``."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self._settings = settings
        self._embedder = embedder if embedder is not None else OllamaEmbedder(settings)

    def _store(self) -> Store:
        return Store.open(
            self._settings.db_path, self._settings.embed_dim, self._settings.embed_model
        )

    def query(self, text: str, k: int = 5, kind: str | None = None) -> list[dict]:
        """Hybrid (vector + BM25 -> RRF) search; returns ranked, JSON-friendly hits."""
        with self._store() as store:
            results = search(store, self._embedder, text, k=k, kind=kind)
            return [_hit_dict(r.hit, score=r.score) for r in results]

    def get(self, chunk_id: int) -> dict | None:
        """Full text + location for one chunk, or ``None`` if it is gone/tombstoned."""
        with self._store() as store:
            hit = store.get_chunk(chunk_id)
        return None if hit is None else _hit_dict(hit, score=None, full=True)

    def status(self) -> dict:
        """Index stats + backend health, for ops and the MCP ``status`` tool."""
        with self._store() as store:
            index = store.index_stats()
        return {
            "index": index,
            "embed_model": self._settings.embed_model,
            "ollama_url": self._settings.ollama_url,
            "ollama_ok": self._ping(),
        }

    def _ping(self) -> bool:
        try:
            resp = httpx.get(f"{self._settings.ollama_url.rstrip('/')}/api/version", timeout=2.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _hit_dict(hit: SearchHit, *, score: float | None, full: bool = False) -> dict:
    text = hit.text if full else hit.text[:MAX_TEXT_CHARS]
    out = {
        "chunk_id": hit.chunk_id,
        "kind": hit.kind,
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "symbol": hit.symbol,
        "text": text,
    }
    if score is not None:
        out["score"] = round(score, 6)
    return out
