# qmx — ML Notes & Technical Debt

Model/ML-side decisions, deferrals, and things to revisit. Companion to the code; when a shortcut is
taken for pragmatism, it gets an entry here so it isn't lost.

## Technical debt

### TD-1 · Reranker deferred — RRF-only ranking (Phase 3)

**Status:** deferred (2026-07-16). Ranking is **vector + BM25 → Reciprocal Rank Fusion**, no rerank
stage. A `Reranker` seam exists (`qmx/rerank.py`, `search(..., reranker=...)`) with a no-op default,
so a real reranker can be slotted in without touching call sites.

**Why deferred — what we found on the Spark (Ollama 0.32.1):**
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
