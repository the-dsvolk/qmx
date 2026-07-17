# qmx — ML Notes & Technical Debt

Model/ML-side decisions, deferrals, and things to revisit. Companion to the code; when a shortcut is
taken for pragmatism, it gets an entry here so it isn't lost.

## Technical debt

### TD-1 · Reranker — ~~deferred~~ **RESOLVED** (Qwen3-Reranker via llama.cpp on the GB10)

**Status:** **resolved (2026-07-17).** Reranking is a real cross-encoder — **Qwen3-Reranker-0.6B**
served by **llama.cpp `llama-server --reranking`** on the Spark **GPU**, called by
`qmx/rerank.py:HttpReranker` behind the existing seam. Off unless `rerank_url` is set (still RRF-only
by default); **fails soft** to RRF order if the server is unreachable.

**How it was done (the path that worked after Ollama and TEI both didn't):**
- Ollama has no rerank endpoint (see below); TEI's images are **amd64-only** (Spark is aarch64) and
  Docker has no nvidia runtime → neither runs here.
- **llama.cpp built from source, natively on aarch64, with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`**
  (`nvcc` from `/usr/local/cuda-13.0`, which lists `compute_121`). GGML CUDA is *proven* on the GB10
  because Ollama already runs on it — that de-risked the build.
- Model: **`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`** (published by the llama.cpp org — confirms
  `--reranking` supports Qwen3-Reranker). ~610 MB, offloads to the GB10 (`-ngl 99`, ~5.6 GiB VRAM).
- Served as a **`systemd --user` service** `qmx-rerank.service` bound `0.0.0.0:8081`, reboot-safe
  (linger + `WantedBy=default.target` + `Restart=on-failure`). Endpoint: `/v1/rerank` (Cohere-style
  `{"results":[{"index","relevance_score"}]}`).
- **Gotcha:** default `n_ubatch=512` rejects our real code chunks (a single query+doc can exceed 512
  tokens → HTTP 500). Fixed with **`-b 8192 -ub 8192`** (n_ctx is 40960; ample VRAM).
- qmx side: `HttpReranker` (thin HTTP client), `Settings.rerank_url` (+ `QMX_RERANK_URL`), wired
  through `search(reranker=...)` and `QmxService` (so MCP `query`/`search_code`/`recall` rerank);
  `search` reranks the RRF top-`rerank_pool` (40) then trims to `k`.
- Verified end-to-end: `query "ssh into a kubernetes pod"` → `ssh_to_pod`/`ssh_to_job` with
  cross-encoder scores ~0.999; irrelevant docs ~1e-5.

To enable on a client: `rerank_url = "http://spark-0e81.local:8081"` in `~/.qmx/config.toml`.

---

**Historical context — why Ollama/TEI didn't work (Spark, Ollama 0.32.1):**
- `GET/POST /api/rerank` → **404**. Ollama has **no rerank endpoint**.
- `ollama pull qwen3-reranker[:0.6b]` → **"file does not exist"** — not in the Ollama library.
- So the plan's assumption (Qwen3-Reranker served by Ollama, thin HTTP client) is **not achievable**
  as written. `/api/embed` works fine (embeddings are unaffected).

RRF-only already performs well in practice: the Phase 1 Spark validation returned the correct
function at **rank 1** for by-meaning queries over qmx's own source. Reranking is a precision
refinement, not a prerequisite — hence safe to defer.

**Options when we revisit (pick per quality need vs. stack cost):**
1. **LLM-as-reranker via the Qwen chat model** (Ollama `generate`, no torch). Pointwise/listwise
   relevance scoring over the RRF top-k. Keeps the thin-client / no-torch stack; ~one extra chat
   call per query. Not a true cross-encoder but improves ordering. *Leading candidate.*
2. **Dedicated rerank server** exposing `/rerank` (e.g. HuggingFace text-embeddings-inference or
   `infinity`) running a real cross-encoder (bge-reranker / Qwen3-Reranker) next to Ollama. qmx stays
   a thin HTTP client — just a second backend URL. Best quality without torch in qmx.
3. **In-process Qwen3-Reranker** (transformers + torch on the Spark). Highest fidelity to the
   original plan, but pulls in torch and couples the daemon to the GPU box — violates the
   "thin HTTP client, no torch" decision in [qmx-plan.md](./qmx-plan.md) / [qmx-deployment.md](./qmx-deployment.md).

**Revisit trigger:** when top-5 ordering quality becomes a felt problem, or once chats (Phase 4) are
indexed and cross-domain ranking needs sharpening. Acceptance to reclaim: rerank measurably improves
top-5 ordering on a small labelled query set.

**Touch points:** `qmx/rerank.py` (protocol + `NoOpReranker`), `qmx/search.py` (`reranker` param),
`Settings.rerank_model` / `chat_model` (config seam already present).
