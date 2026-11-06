# Reranker: Qwen3-Reranker-0.6B → llama-nemotron-rerank-1b-v2 (eval 2026-11-06)

**Result:** `nvidia/llama-nemotron-rerank-1b-v2` (rev `82876565`) beats the production
Qwen3-Reranker-0.6B on every metric, with every 95% CI excluding zero, and is faster. Production
is unchanged for now; switching is a separate decision (see "Before switching").

This is the follow-up the embedder eval called for (`plan/qmx-embed-nemotron.md`, "the Qwen
reranker is now the bottleneck").

## Method

- **Same harness, same labels.** `scripts/embed_eval.py` (`retrieve` / `label` / `report`) with the
  embedder eval's known-item and real-query sets. Rerank latency is now recorded per query.
- **One frozen snapshot.** `sqlite3 .backup` of the live index (85,418 chunks, Nemotron-3-Embed-1B @1024).
  Each arm gets a byte-identical copy, because qmx sets WAL mode on open and two processes on one file
  collide. Both configs equal production except `rerank_url` / `rerank_model`.
- **Paired check.** Vector and RRF ranks are identical in both arms on 336/336 known-item queries, so
  the reranker is the only difference.
- **Known-item:** 336 of the original 340 queries. Two code chunks lost their document link since
  the embedder eval, so their 4 queries are dropped. **Real queries:** the same 40; the production
  judge graded the 94 newly surfaced (query, chunk) pairs (297 pooled in total).
- **Servers.** Qwen: production `qmx-rerank.service` (llama.cpp, Q8_0) on spark-0e81:8081.
  Nemotron: vLLM v0.33.0 container `nemo-rerank` on spark-0f50:8003 (`--runner pooling
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.05`). It must have the
  model card's score template (`--chat-template`, `question:{q} \n \n passage:{p}`); without it
  the scores are wrong. Smoke test reproduced the card's example logits (−23.08 vs −23.1).
- Production reranks the RRF top 40 and trims to 5; unchanged here.

## Results

**Known-item retrieval (336 queries, full pipeline)**

| reranker | Recall@1 | Recall@5 | MRR@5 |
|---|---|---|---|
| none (RRF only) | 0.500 | 0.801 | 0.614 |
| Qwen3-Reranker-0.6B | 0.491 | 0.812 | 0.614 |
| **llama-nemotron-rerank-1b-v2** | **0.640** | **0.905** | **0.747** |

Nemotron − Qwen (paired bootstrap, 10k): Recall@5 **+0.092 [+0.057, +0.131]**, MRR@5
**+0.133 [+0.095, +0.172]**.

Full-pipeline MRR@5 by group (Qwen → Nemotron): chat 0.491 → 0.644, code 0.728 → 0.816,
doc 0.644 → 0.749, learning 0.726 → 0.826, meeting 0.494 → 0.707, memory 0.471 → 0.733;
keyword queries 0.625 → 0.718, paraphrased questions 0.603 → 0.775.

**Real past queries (40, judge-graded 0/1/2, pooled):** nDCG@5 0.920 → **0.973**
(+0.053 [+0.020, +0.091]); share of top-5 graded 2: 0.885 → 0.935.

**Latency (RRF top-40 reranked, per query):** Qwen median 678 ms / p90 940 ms; Nemotron median
579 ms / p90 804 ms. Not a clean comparison: the boxes differ, and Nemotron shared spark-0f50's GPU
with an actively generating Nemotron 3.5 Super for most of its run.

**Memory:** Nemotron ~6 GB (5% budget) vs Qwen's 10.7 GB (llama.cpp reserves the full 40,960-token
context).

## What it means

- The Qwen reranker adds almost nothing on top of Nemotron embeddings + BM25 (Recall@5 +0.011,
  Recall@1 −0.009). The NVIDIA reranker adds +0.104 Recall@5 and +0.140 Recall@1 to the same
  candidates.
- The Qwen judge wrote the queries and graded the results, so any family bias favours the
  incumbent; the margin is conservative.

## Before switching

1. **Cap the input length.** The model takes at most 8,192 tokens per query+passage; vLLM rejects an
   over-long request (HTTP 400), and `HttpReranker` would then silently keep RRF order for that
   query. Today's longest chunk is 23,240 chars (~6k tokens), so it fits now. Send
   `truncate_prompt_tokens` or trim passages client-side so this can't happen.
2. Run `nemo-rerank` on the primary spark with `--restart unless-stopped` and `HF_HUB_OFFLINE=1`
   (same flags and template as above), point `rerank_url` / `rerank_model` at it, then stop and
   disable `qmx-rerank.service`. Keep the GGUF and unit file for rollback.
