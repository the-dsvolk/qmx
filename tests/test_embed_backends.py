"""Embedding backends: OpenAI-compatible client (vLLM), prefixes, the query side, config keys."""

from __future__ import annotations

import base64
import json
import math
from array import array
from pathlib import Path

import httpx
import pytest

from qmx import embed as embed_mod
from qmx.config import Settings
from qmx.embed import (
    CircuitBreakerEmbedder,
    EmbedBackendError,
    OllamaEmbedder,
    OpenAIEmbedder,
    embed_queries,
    make_embedder,
    ping_embedder,
)
from qmx.index import index_paths
from qmx.search import search
from qmx.store import Store
from tests.fakes import FakeEmbedder

NEMO = dict(
    embed_backend="openai",
    embed_url="http://embed.test/v1",
    embed_model="nemotron-3-embed-1b",
    embed_query_prefix="query: ",
    embed_passage_prefix="passage: ",
    embed_max_tokens=8192,
    max_retries=1,
)


def _b64(vec: list[float]) -> str:
    return base64.b64encode(array("f", vec).tobytes()).decode()


def _openai(settings: Settings, vectors_for, seen: list[dict]) -> OpenAIEmbedder:
    """An OpenAIEmbedder whose server answers ``vectors_for(inputs)`` and records each request."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"path": request.url.path, **body})
        vectors = vectors_for(body["input"])
        data = [{"index": i, "embedding": _b64(v)} for i, v in enumerate(vectors)]
        return httpx.Response(200, json={"data": list(reversed(data))})  # order must not matter

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.embed_url)
    return OpenAIEmbedder(settings, client=client)


def test_openai_embedder_sends_prefixes_base64_and_truncation():
    seen: list[dict] = []
    s = Settings(**NEMO, embed_dim=4)
    emb = _openai(s, lambda inputs: [[1.0, 0.0, 0.0, 0.0]] * len(inputs), seen)
    assert emb.embed(["a chunk"]) == [[1.0, 0.0, 0.0, 0.0]]
    assert emb.embed_query(["a question"]) == [[1.0, 0.0, 0.0, 0.0]]
    assert [r["input"] for r in seen] == [["passage: a chunk"], ["query: a question"]]
    assert seen[0]["path"] == "/v1/embeddings"
    assert seen[0]["model"] == "nemotron-3-embed-1b"
    assert seen[0]["encoding_format"] == "base64"
    assert seen[0]["truncate_prompt_tokens"] == 8192


def test_openai_embedder_keeps_input_order_across_batches():
    seen: list[dict] = []
    s = Settings(**{**NEMO, "embed_query_prefix": "", "embed_passage_prefix": ""}, embed_dim=2,
                 embed_batch_size=2)
    emb = _openai(s, lambda inputs: [[float(t), 1.0] for t in inputs], seen)
    out = emb.embed(["1", "2", "3"])
    assert [round(v[0] / v[1]) for v in out] == [1, 2, 3]
    assert len(seen) == 2  # batched 2 + 1


def test_openai_embedder_cuts_wider_vectors_and_renormalises():
    s = Settings(**NEMO, embed_dim=2)
    emb = _openai(s, lambda inputs: [[3.0, 4.0, 12.0, 0.0]] * len(inputs), [])
    [vec] = emb.embed(["x"])
    assert len(vec) == 2
    assert math.isclose(math.hypot(*vec), 1.0, rel_tol=1e-6)
    assert vec == pytest.approx([0.6, 0.8])


def test_openai_embedder_rejects_narrower_vectors():
    s = Settings(**NEMO, embed_dim=8)
    emb = _openai(s, lambda inputs: [[1.0, 0.0]] * len(inputs), [])
    with pytest.raises(EmbedBackendError, match="< configured embed_dim 8"):
        emb.embed(["x"])


def test_openai_embedder_bad_body_is_a_backend_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": []})),
        base_url="http://embed.test/v1",
    )
    with pytest.raises(EmbedBackendError, match="unexpected response body"):
        OpenAIEmbedder(Settings(**NEMO, embed_dim=2), client=client).embed(["x"])


def test_openai_embedder_needs_a_url():
    with pytest.raises(ValueError, match="needs embed_url"):
        OpenAIEmbedder(Settings(embed_backend="openai"))


def test_ollama_embedder_sends_text_unchanged_by_default_and_prefixes_when_set():
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        seen.append(inputs)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(inputs)})

    def make(**kw) -> OllamaEmbedder:
        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://o.test")
        return OllamaEmbedder(Settings(embed_dim=2, **kw), client=client)

    make().embed(["doc"])
    make().embed_query(["q"])
    make(embed_query_prefix="Instruct: find\nQuery: ").embed_query(["q"])
    assert seen == [["doc"], ["q"], ["Instruct: find\nQuery: q"]]


def test_make_embedder_selects_the_backend():
    assert isinstance(make_embedder(Settings()), OllamaEmbedder)
    assert isinstance(make_embedder(Settings(**NEMO)), OpenAIEmbedder)
    with pytest.raises(ValueError, match="embed_backend must be one of"):
        make_embedder(Settings(embed_backend="tei"))


class _Sides(FakeEmbedder):
    """Records which side each call used."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim)
        self.calls: list[str] = []

    def embed(self, texts):
        self.calls.append("passage")
        return super().embed(texts)

    def embed_query(self, texts):
        self.calls.append("query")
        return super().embed(texts)


def test_embed_queries_uses_query_side_or_falls_back():
    sides = _Sides()
    embed_queries(sides, ["q"])
    assert sides.calls == ["query"]
    plain = FakeEmbedder()
    assert embed_queries(plain, ["q"]) == plain.embed(["q"])


def test_circuit_breaker_routes_query_side_and_trips_on_it():
    sides = _Sides()
    cb = CircuitBreakerEmbedder(sides)
    cb.embed(["d"])
    cb.embed_query(["q"])
    assert sides.calls == ["passage", "query"]

    class Down(FakeEmbedder):
        def embed_query(self, texts):
            raise EmbedBackendError("down")

    cb = CircuitBreakerEmbedder(Down(), cooldown=60)
    with pytest.raises(EmbedBackendError):
        cb.embed_query(["q"])
    assert cb.degraded


def test_search_embeds_the_query_with_the_query_side(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "net.py").write_text("def retry_with_backoff(func):\n    return func()\n")
    sides = _Sides()
    with Store.open(tmp_path / "index.db", sides.dim, "fake") as store:
        index_paths([repo], store, sides)
        assert set(sides.calls) == {"passage"}  # indexing is document side only
        sides.calls.clear()
        hits = search(store, sides, "retry backoff", k=3)
    assert sides.calls == ["query"]
    assert hits and hits[0].hit.path.endswith("net.py")


def test_ping_embedder_hits_the_configured_backend(monkeypatch):
    urls: list[str] = []

    def fake_get(url, timeout):
        urls.append(url)
        return httpx.Response(200)

    monkeypatch.setattr(embed_mod.httpx, "get", fake_get)
    assert ping_embedder(Settings(**NEMO))
    assert ping_embedder(Settings(ollama_url="http://o.test:11434/"))
    assert urls == ["http://embed.test/v1/models", "http://o.test:11434/api/version"]


def test_embed_settings_from_toml_and_env(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'embed_backend = "openai"\n'
        'embed_url = "http://spark-0f50.local:8002/v1"\n'
        'embed_query_prefix = "query: "\n'
        'embed_passage_prefix = "passage: "\n'
    )
    s = Settings.load(
        config_path=cfg,
        env={"QMX_EMBED_MAX_TOKENS": "8192", "QMX_PARKED_SIMILAR_MAX_DISTANCE": "0.42"},
    )
    assert (s.embed_backend, s.embed_url) == ("openai", "http://spark-0f50.local:8002/v1")
    assert (s.embed_query_prefix, s.embed_passage_prefix) == ("query: ", "passage: ")
    assert s.embed_max_tokens == 8192
    assert s.parked_similar_max_distance == 0.42
    d = Settings.load(config_path=Path("/does/not/exist"), env={})
    assert (d.embed_backend, d.embed_url, d.embed_query_prefix, d.embed_passage_prefix) == (
        "ollama", "", "", "")
    assert (d.embed_max_tokens, d.parked_similar_max_distance) == (0, 0.3)
