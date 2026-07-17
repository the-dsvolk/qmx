"""Live Ollama round-trip — skipped unless a backend is reachable.

Run against the Spark:  QMX_OLLAMA_URL=http://spark-0e81.local:11434 uv run pytest -m integration
"""

from __future__ import annotations

import httpx
import pytest

from qmx.config import Settings
from qmx.embed import OllamaEmbedder
from qmx.store import Chunk, Store, hash_text

pytestmark = pytest.mark.integration


def _backend_up(url: str) -> bool:
    try:
        return httpx.get(f"{url.rstrip('/')}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture
def settings():
    s = Settings.load()
    if not _backend_up(s.ollama_url):
        pytest.skip(f"no Ollama backend at {s.ollama_url}")
    return s


def test_live_embed_and_search(settings, tmp_path):
    with OllamaEmbedder(settings) as embedder:
        texts = ["retry logic with exponential backoff", "sqlite full text search", "grace hopper"]
        vectors = embedder.embed(texts)
        assert len(vectors) == 3
        assert all(len(v) == settings.embed_dim for v in vectors)

        with Store.open(tmp_path / "index.db", settings.embed_dim, settings.embed_model) as store:
            doc_id = store.upsert_document(kind="code", path="live.py")
            embeddings = {hash_text(t): v for t, v in zip(texts, vectors, strict=True)}
            store.reindex_document(doc_id, [Chunk(text=t) for t in texts], embeddings)
            [qvec] = embedder.embed(["how do I retry a failed request?"])
            hits = store.search_vec(qvec, k=3)
            assert hits[0].text == "retry logic with exponential backoff"
