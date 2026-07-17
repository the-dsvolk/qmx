# qmx — Deployment & Dev Cycle

Where qmx **runs**, where you **develop** it, and how the two machines (Mac + DGX Spark) split the
work. This resolves the "dev on Mac vs run on Spark" question. It complements
[qmx-plan.md](./qmx-plan.md) (what to build) and [qmx-architecture.md](./qmx-architecture.md)
(how the capabilities work) — this doc is purely about *topology*.

## Decision (one line)

**The Spark is a resident qmx service (Ollama + unified index + HTTP MCP); the Mac is a dev box +
thin client. Develop in two tiers split by GPU-need; use Remote-SSH for the model tier.**

## Why the split

Three forces pull in different directions:

- **The models want the Spark.** The Qwen stack (embed / rerank / summarize) is the compute-heavy
  part. GB10 + 128 GB unified memory chews through a full-corpus reindex; the Mac (Metal/MLX) is
  fine for a toy index, not for "many repos + years of transcripts." The Spark is the natural
  **index + model engine**.
- **The data is split.** Code repos and Claude Code transcripts (`~/.claude/projects/*.jsonl`) live
  wherever you work — some on the Mac, some on the Spark. The indexer must reach both.
- **MCP plugs into Claude Code on *both* machines** → one server both can reach, i.e. **HTTP**, not
  a per-machine stdio server.

Conclusion: not a "build here, ship there" pipeline — a **resident service on the Spark** with the
Mac as a dev box + client.

## Topology

```
┌─ Mac (dev + client) ──────┐        ┌─ Spark spark-0e81 (the qmx service) ────┐
│ • edit code               │  LAN   │ • Ollama (Qwen, CUDA)  :11434 localhost  │
│ • Claude Code ── MCP/HTTP ─┼───────►│ • qmx daemon: index / watch / consolidate│
│ • Stop hook POSTs turns ──┼───────►│ • SQLite + sqlite-vec + FTS5  (the index)│
│ • local repo checkouts    │        │ • MCP HTTP server :PORT  ◄── both boxes  │
└───────────────────────────┘        │ • Claude Code here ── MCP/localhost      │
                                      └──────────────────────────────────────────┘
```

- **One unified index on the Spark** — single source of truth, GPU-adjacent, on fast NVMe.
- **Both Claude Codes query the same server** — Mac over `http://spark-0e81.local:PORT`, Spark over
  localhost.
- **Privacy scope shifts** from "never leaves the machine" to "never leaves my LAN / my two
  devices." Mac data now transits to the Spark over the LAN — a conscious, accepted trade.

## Forks resolved

| # | Fork | Decision | Rationale |
|---|---|---|---|
| 1 | Model serving | **Ollama on the Spark** (keeps [qmx-plan](./qmx-plan.md)'s thin-client / no-torch decision; only the *host* moves Mac→Spark) | The `OLLAMA_URL` is the config seam; app stays torch-free and runs identically on both boxes. In-process kept only as a fallback if Ollama's Qwen3-Reranker support proves weak. |
| 2 | MCP transport | **One HTTP MCP server** for both machines | Resident background loops (capture/consolidate) must outlive any Claude Code session; HTTP lets the Mac reach it too. |
| 3 | Index location | **Single index on the Spark** (`~/.qmx/index.db` on `spark-0e81`) | One flat KB (already decided); GPU-adjacent; no cross-machine merge. |
| 4 | Mac data → Spark | **Chats: Stop-hook live POST** to the Spark endpoint. **Code: Remote-SSH / mutagen** (canonical checkout on the Spark). | The hook POSTing turns gives one unified chat index no matter where you chatted — cleaner than syncing JSONL. |

## The config seam (the thing that makes it all work)

Two settings, resolved per-machine from env/TOML, let the *same code* run in three modes with no
rewrites:

- `QMX_OLLAMA_URL` — where the models are. Mac dev → `http://spark-0e81.local:11434`; Spark →
  `http://localhost:11434`; CI → a mock/tiny embedder.
- `QMX_MCP_URL` — where the daemon is. Both Claude Codes point their MCP client here
  (`http://spark-0e81.local:PORT`; localhost on the Spark).

| Mode | `QMX_OLLAMA_URL` | Use |
|---|---|---|
| CI / unit test | mock / tiny embedder | fast, deterministic, no GPU |
| Mac local | `spark-0e81.local:11434` (or a small local Qwen) | quick end-to-end sanity |
| Spark prod | `localhost:11434` | the real thing (CUDA) |

## Dev cycle — two tiers, split by GPU-need

The trap: developing model code on the Mac (Metal/MLX) and shipping to the Spark (CUDA) is the
classic "works on my machine." Avoid it by splitting **where you develop** by **what the code
touches**:

- **CPU-only logic → develop + unit-test on the Mac.** `chunk/*`, `store.py` (SQLite/FTS/vec
  schema), `search.py` RRF fusion, `mcp_server.py` protocol plumbing, `capture.py` cleaning, hooks.
  Fast, no GPU, tight `pytest` loop.
- **Model-touching + throughput + recall quality → develop on the Spark directly.** `embed.py`,
  the reranker stage, indexing speed, "does search actually find the right thing." Get CUDA parity
  by editing where it runs.

Spark-tier inner loop (pick one):

1. **Cursor / VS Code Remote-SSH into `spark-0e81.local`** — edit with local UX, runs where edited,
   no sync step. Recommended for the model tier.
2. **`mutagen` continuous sync** Mac→Spark, run over SSH — keep the Mac editor, near-instant
   propagation.
3. Plain **rsync-on-save** — crude, zero deps.

Reserve **git push/pull for checkpoints**, not the inner loop. The Spark already has an rw deploy
key (`github-qmx` alias) so it can commit and push too. `uv` on both machines → identical envs.

## Prod / run on the Spark

- **`systemd --user` services** for Ollama + the qmx daemon → auto-start, restart-on-crash,
  `journalctl` logs. `loginctl enable-linger dsvolk` so they run without an active login session.
- **Incremental reindex on a timer** (per-file / per-chunk hashing, already in the plan) + a
  `status()` MCP call for health (index stats, last index time, Ollama reachable).
- **Graceful degradation** while Ollama loads / is down — queue + backoff (already a stated goal).

## Open (deployment-specific)

1. **MCP over the LAN** — direct HTTP from the Mac's Claude Code vs. a thin local stdio→HTTP bridge.
   (Lean: direct HTTP; bridge only if a client needs stdio.)
2. **Endpoint exposure** — bind the daemon to `spark-0e81.local` (LAN) only; no auth needed on a
   trusted LAN, but confirm the bind address isn't `0.0.0.0` on an untrusted network.
3. **Reranker backend** — validate Ollama's Qwen3-Reranker throughput/quality in Phase 3; fall back
   to in-process reranking only if it's inadequate.
