"""Phase 0 acceptance: embed 3 strings, store, cosine top-k returns them."""

from __future__ import annotations

import pytest

from qmx.store import Chunk, Store, StoreSchemaMismatch
from tests.fakes import FakeEmbedder

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "database vector search engine with cosine similarity",
    "python asyncio http client with retry and backoff",
]


@pytest.fixture
def store(tmp_path):
    embedder = FakeEmbedder(dim=64)
    with Store.open(tmp_path / "index.db", embedder.dim, "fake") as s:
        yield s, embedder


def test_roundtrip_topk_returns_the_stored_string(store):
    s, embedder = store
    doc_id = s.upsert_document(kind="code", path="sample.py")
    chunks = [Chunk(text=t, ord=i) for i, t in enumerate(TEXTS)]
    s.add_chunks(doc_id, chunks, embedder.embed(TEXTS))

    assert s.counts() == {"documents": 1, "chunks": 3, "vectors": 3}

    for query in TEXTS:
        [qvec] = embedder.embed([query])
        hits = s.search_vec(qvec, k=3)
        assert len(hits) == 3
        assert hits[0].text == query  # exact match ranks first
        assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
        assert hits[0].distance <= hits[1].distance <= hits[2].distance


def test_dedup_identical_chunk_embeds_once(store):
    s, embedder = store
    doc_id = s.upsert_document(kind="code", path="dup.py")
    dup = [Chunk(text="same text"), Chunk(text="same text")]
    ids = s.add_chunks(doc_id, dup, embedder.embed([c.text for c in dup]))
    assert ids[0] == ids[1]
    assert s.counts()["chunks"] == 1


def test_kind_filter(store):
    s, embedder = store
    code_doc = s.upsert_document(kind="code", path="a.py")
    chat_doc = s.upsert_document(kind="chat", path="b.md")
    s.add_chunks(
        code_doc, [Chunk(text="vector search in code")], embedder.embed(["vector search in code"])
    )
    s.add_chunks(
        chat_doc, [Chunk(text="vector search in chat")], embedder.embed(["vector search in chat"])
    )

    [qvec] = embedder.embed(["vector search"])
    chat_hits = s.search_vec(qvec, k=5, kind="chat")
    assert chat_hits and all(h.kind == "chat" for h in chat_hits)


def test_upsert_document_is_idempotent(store):
    s, _ = store
    first = s.upsert_document(kind="code", path="x.py", file_hash="aaa")
    second = s.upsert_document(kind="code", path="x.py", file_hash="bbb")
    assert first == second
    assert s.counts()["documents"] == 1


def test_reopen_with_wrong_dim_raises(tmp_path):
    path = tmp_path / "index.db"
    Store.open(path, 64, "fake").close()
    with pytest.raises(StoreSchemaMismatch):
        Store.open(path, 128, "fake")
