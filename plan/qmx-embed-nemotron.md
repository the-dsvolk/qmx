# Embeddings: Qwen3-Embedding-0.6B → Nemotron-3-Embed-1B (eval 2026-11-05)

**Decision:** switch the embedder to `nvidia/Nemotron-3-Embed-1B-BF16` (rev `c0c9fea9`) at
**1024 dims**, with `query: ` / `passage: ` prefixes, `dedup_min_similarity = 0.50` and
`parked_similar_max_distance = 0.45`. The reranker (Qwen3-Reranker-0.6B) and judge stay as they are.

The bar was set before any results: the full production pipeline must be at least as good as Qwen
and better on a main metric with a 95% CI excluding zero, and duplicate separation must be no worse.
It passed. The production-level gain is modest; the embedder itself is far better, but keyword search
and the reranker mask most of that today (see "What it means").

## Method

- **Identical snapshots.** One `sqlite3 .backup` of the live index (83,363 chunks), copied three
  times; `qmx reembed` rebuilt only `vec_chunks` in the copies (Nemotron 2048 dims: 566 s, 1024: 556 s,
  ~147 chunks/s on the second spark). Text, keyword index, mentions and lessons are byte-identical.
- **Production code paths.** `qmx.search.search` with the production reranker; `store.search_vec`
  for the vector-only arm. A failed embed or rerank aborts the run (none occurred).
- **Neutral-to-conservative labels.** The production judge (Qwen3.8-27B) wrote the queries and graded
  results, so any family bias favours the incumbent Qwen embedder.
- **Test sets** (harness: `scripts/embed_eval.py`; data stays in the private `--dir`):
  1. *Known-item:* 175 chunks sampled by kind (lessons 50, chats 50, code 25, docs 25, meetings 15,
     memory 10); the judge wrote a 2–6 word keyword query and a paraphrased question for each
     (340 usable). Target = the source chunk. Keyword queries share 80% of their words with the
     passage, questions 40%.
  2. *Real queries:* 40 sampled from 79 distinct past `mcp__qmx__query` calls; pooled top-5 of every
     setup graded 0/1/2 by the judge (239 judgements); nDCG@5.
  3. *Duplicate lessons:* 100 random live lessons × top-3 neighbours under each model, labelled with
     the production dedup prompt (filing rules v2): 490 pairs, 36 judged the same lesson.
- Paired bootstrap (10k resamples) for retrieval; stratified bootstrap for AUC.

## Results

**Known-item retrieval (340 queries)**

| setup | arm | Recall@1 | Recall@5 | MRR@5 | Recall@20 |
|---|---|---|---|---|---|
| Qwen 0.6B | vector only | 0.244 | 0.447 | 0.320 | 0.635 |
| Qwen 0.6B | vector + BM25 (RRF) | 0.409 | 0.682 | 0.511 | |
| Qwen 0.6B | **full (RRF + rerank)** | 0.476 | 0.791 | 0.600 | |
| Nemotron 1024 | vector only | 0.497 | 0.803 | 0.608 | 0.929 |
| Nemotron 1024 | vector + BM25 (RRF) | 0.500 | 0.806 | 0.616 | |
| Nemotron 1024 | **full (RRF + rerank)** | 0.482 | **0.826** | 0.613 | |
| Nemotron 2048 | vector only | 0.524 | 0.824 | 0.639 | 0.944 |
| Nemotron 2048 | full (RRF + rerank) | 0.474 | 0.826 | 0.609 | |

Nemotron 1024 − Qwen (95% CI): vector Recall@5 **+0.356 [+0.303, +0.409]**; RRF Recall@5
**+0.124 [+0.082, +0.165]**; full Recall@5 **+0.035 [+0.006, +0.065]**; full MRR@5 +0.013
[−0.006, +0.033] (n.s.). Full-pipeline MRR@5 by kind: ahead on chats, code, docs, meetings; level on
lessons (0.70 vs 0.72) and memory files (20 queries).

**Real past queries (40):** nDCG@5 Qwen 0.948, Nemotron 1024 0.947 (−0.001 [−0.022, +0.024]),
2048 0.941. Near the ceiling — agent-written queries are keyword-rich, so BM25 + rerank already find
the target — so this set cannot separate the models; it shows no regression.

**Duplicate lessons (490 pairs, 36 duplicates)**

| setup | duplicates in its top-3 | AUC | look-alikes passing at the cut keeping 90% of duplicates |
|---|---|---|---|
| Qwen 0.6B | 27/36 | 0.672 | 86% |
| Nemotron 1024 | 28/36 | 0.689 | 78% |
| Nemotron 2048 | 29/36 | 0.707 | 63% |

AUC difference vs Qwen: 1024 +0.017 [−0.075, +0.107], 2048 +0.035 [−0.058, +0.128] — no worse, not
significantly better with 36 positives.

**Latency (median per query, quiet servers):** Qwen query embed 191 ms + vector search 120 ms;
Nemotron 1024: 28 ms + 103 ms; Nemotron 2048: 27 ms + 193 ms.

## Thresholds (model-specific — the old values are wrong for Nemotron)

Nemotron cosine similarities run lower than Qwen's. Kept at 0.65, the dedup gate would see only 19%
of true duplicates (Qwen today: 61%).

| setting | Qwen (today) | Nemotron 1024 | how |
|---|---|---|---|
| `dedup_min_similarity` | 0.65 → 61% of duplicates reach the dedup question, 31% of look-alikes | **0.50** → 61% / 31% | same duplicate recall on the labelled pairs |
| `parked_similar_max_distance` | 0.30 | **0.45** | same share of neighbour pairs counted similar |

## What it means

- The embedder is roughly twice as good on its own (vector Recall@5 0.45 → 0.80, Recall@20 0.64 →
  0.93), and faster.
- In the production pipeline the gain shrinks to +3.5 points Recall@5 because **the Qwen reranker is
  now the bottleneck**: for Qwen it lifts MRR@5 from 0.32 (vector only) to 0.60; for Nemotron it
  adds a little Recall@5 (0.803 → 0.826) but costs rank-1 accuracy (Recall@1 0.497 → 0.482; at 2048
  dims 0.524 → 0.474). Next experiment: an NVIDIA reranker (e.g. `llama-nemotron-rerank-1b-v2`) on
  the same harness.
- 1024 vs 2048: identical in the full pipeline, 2048 slightly better vector-only and in dedup, at
  twice the storage and ~90 ms more vector search. 1024 keeps the table the size it is today.

## Switchover runbook

Prereqs: this PR merged; the embedder running where production wants it, with
`--restart unless-stopped` (the eval container `nv-embed3` on `spark-0f50:8002` uses `--restart no`).

```bash
uv tool upgrade qmx                                    # prod qmx installs from GitHub main
cp ~/.qmx/config.toml ~/.qmx/config.toml.bak-$(date +%Y%m%d)-nemo-embed
for s in serve watch consolidate; do launchctl bootout gui/$UID ~/Library/LaunchAgents/com.qmx.$s.plist; done
sqlite3 ~/.qmx/index.db ".backup ~/.qmx/index-nemo.db"          # services stopped: final snapshot
QMX_CONFIG=~/.qmx/eval-nemo/nemo1024.toml \
  qmx reembed --db ~/.qmx/index-nemo.db --reuse ~/.qmx/eval-nemo/nemo1024.db   # only new chunks
# config.toml: embed_backend/embed_url/embed_model/embed_dim=1024/embed_*_prefix/embed_max_tokens,
#   dedup_min_similarity = 0.50, parked_similar_max_distance = 0.45
mv ~/.qmx/index.db ~/.qmx/index-qwen.db && mv ~/.qmx/index-nemo.db ~/.qmx/index.db
for s in serve watch consolidate; do launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.qmx.$s.plist; done
qmx status            # embed_model nemotron-3-embed-1b, embed_ok true, vectors == chunks
```

Rollback: restore the config backup, `mv` the DB names back, restart the three services. Keep the
Ollama Qwen embedder on the primary spark until the new one has run for a week or two.
