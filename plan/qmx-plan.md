# qmx — Implementation Plan

**qmx = Query Memory indeX** — local, private, Qwen-powered semantic search over **code *and* chats**.

## Decision record

- **Approach:** from-scratch **Python 3.12 / `uv`**, clean-room (not a qmd fork). Derived conceptually
  from `tobi/qmd` (MIT); no TS carried over.
- **Model hosting:** **Ollama** serves all models. qmx is a thin HTTP client (`QMX_OLLAMA_URL`) —
  no in-process model loading, no torch dependency. **Ollama runs on the DGX Spark** (GB10/CUDA),
  not the Mac; the Mac dev client points at `spark-0e81.local:11434`. See
  [qmx-deployment.md](./qmx-deployment.md) for the full Mac-dev / Spark-prod topology.
- **Models — Qwen only** (one Apache-2.0 family):
  - `qwen3-embedding` — embeddings (vector search), served by Ollama (`/api/embed`)
  - `qwen3-reranker` — final-stage reranking, served by **llama.cpp `llama-server --reranking`**
    (Ollama has no rerank endpoint) — see [qmx-ml-notes.md](./qmx-ml-notes.md) TD-1
  - a Qwen chat model (e.g. `qwen3.6:35b-a3b`) — chat-turn summarization / consolidation
- **Interface:** a **resident MCP server** (pay startup once) + a thin CLI for indexing/admin.
- **Store:** SQLite + `sqlite-vec` (vectors) + FTS5 (BM25). Files on disk are the source of truth;
  the DB is a **rebuildable shadow index**.

## Scope: code AND chats are both first-class

qmx is a **single, flat knowledge base** — one index over everything, no project/collection
scoping. Two domains flow into it:

1. **Code** — your local source repos. AST-aware chunking via tree-sitter.
2. **Chats** — Claude Code conversation history:
   - **Backfill:** the ~86 existing `~/.claude/projects/*/*.jsonl` transcripts (one-time import).
   - **On-the-fly:** a Claude Code **Stop hook** captures each new turn live and indexes it.

Queries search the **whole** knowledge base by default. The only distinction we keep is a lightweight
`kind` tag (`code` | `doc` | `chat` | `learning`) — an optional filter so the three capabilities can
route (code search vs memory recall vs learnings), not a scoping boundary. `repo`/`path` are retained
as **display/citation metadata only** (so results show where they came from), not for scoping.

## Architecture

```
                         ┌──────────────────────── qmx (Python, resident) ─────────────────────────┐
  code repos ─┐          │  ingest → chunk → embed(HTTP) → store → search → serve                   │
  chat .md  ──┼──watch──▶│                                                                          │
  Stop hook ──┘          │  chunk/   tree-sitter (code) · md-aware (docs) · jsonl→turn (chats)      │
                         │  embed/   Ollama HTTP client (batched, retried)                          │
                         │  store/   sqlite-vec + FTS5 + hash tables (incremental, dedup)           │
                         │  search/  vector + BM25 → RRF → Qwen3-Reranker                            │
                         │  index/   walk → hash → diff → upsert/tombstone   ← robustness core       │
                         │  capture/ Stop-hook: turn → clean → daily .md → enqueue index            │
                         │  mcp/     resident server: query / get / status                          │
                         └──────────────────────────────────┬───────────────────────────────────────┘
                                                             │ HTTP :11434
                                        ┌────────────────────▼─────────────────────┐
                                        │ Ollama:  qwen3-embedding · qwen3 (chat)    │
                                        │ rerank: qwen3-reranker (llama.cpp, Spark)  │
                                        └────────────────────────────────────────────┘
```

### Package layout

```
src/qmx/
  __init__.py
  config.py        # paths, model names, globs, Ollama URL — from env/TOML
  chunk/
    code.py        # tree-sitter AST chunking (py/ts/go/rust + regex fallback)
    doc.py         # markdown header/code-fence aware
    chat.py        # jsonl → clean turns → chunks (drops tool payloads/system-reminders)
  embed.py         # Ollama /api/embed client: batching, retry/backoff, timeouts
  store.py         # sqlite-vec + FTS5 schema, upsert/delete, hash tables, migrations
  search.py        # vector + BM25 → RRF → optional Qwen3-Reranker
  index.py         # walk sources, hash-diff, incremental reindex, tombstones
  watch.py         # filesystem watcher for code dirs + chat-md dir
  capture.py       # Stop-hook entrypoint (turn → daily md → enqueue)
  mcp_server.py    # resident MCP server (query/search_code/recall/lessons/add_learning/get/status)
  cli.py           # `qmx index|query|watch|serve|backfill-chats|status`
tests/
plan/              # this doc
```

## Store schema (sketch)

- `documents(doc_id, kind, repo, path, mtime, file_hash, ...)` — one row per source file/session.
- `chunks(chunk_id, doc_id, ord, text, chunk_hash, start_line, end_line, symbol)` — dedup on `chunk_hash`.
- `vec_chunks` — `sqlite-vec` virtual table mapping `chunk_id → embedding`.
- `fts_chunks` — FTS5 over `chunks.text` for BM25.
- `meta(schema_version, embed_model, embed_dim, ...)` — drives migrations + rebuild-on-mismatch.

## Chat memory — detail

- **Backfill (`qmx backfill-chats`):** parse each `~/.claude/projects/*/*.jsonl`; keep human/assistant
  message text + concise tool *summaries*; drop raw tool payloads, `system-reminder`, queue/mode
  events. Emit one markdown per session (`## user` / `## assistant`, `<!-- session:UUID -->` anchor)
  into `~/.claude/chat-md/<project>/<date>.md`, then index.
- **On-the-fly (Stop hook → `qmx capture`):** on each turn, read the last turn, clean it, append to
  today's daily file with the session anchor, enqueue an incremental index of that one file. Local
  Qwen embeddings keep this near-real-time.
- **Memory tiers:**
  - **Raw recall** — every turn indexed (full fidelity). Default.
  - **Distilled (optional)** — nightly/lazy Qwen summaries + `PROJECT.md`/`USER.md`-style facts for
    high-signal recall. Complements (does not replace) the existing `~/.claude/.../memory/` system.

## Claude Code triggers (hooks) — how live capture fires

On-the-fly chat capture is driven by **Claude Code hooks**, configured in `settings.json`
(`hooks` block). The harness runs these commands on events and passes JSON on stdin
(`session_id`, `transcript_path`, `cwd`, `hook_event_name`).

- **`Stop`** (primary trigger) — fires when Claude finishes a turn. Command → `qmx capture`, which
  reads `transcript_path` from stdin, extracts the just-completed turn(s), cleans them, appends to
  the daily chat markdown with the session anchor, and enqueues an incremental index of that file.
- **`SubagentStop`** (optional) — same, for subagent turns if we want them captured.
- **`SessionStart`** (optional) — ensure the `qmx watch`/MCP server is up and the index is current.

Config sketch (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "qmx capture" } ] }
    ]
  }
}
```

`qmx capture` is intentionally cheap and non-blocking (enqueue-and-return) so it never slows a turn.
Wiring is done in **Phase 4** via the `update-config` skill (hooks are harness-executed, not model
behavior — they must live in `settings.json`). The MCP server is registered separately (the *read*
door); the Stop hook is the *write* door.

## Robustness core (the reason we chose the "robust" build)

- **Incremental reindex:** per-file `file_hash` skips unchanged files entirely; changed files are
  re-chunked and diffed at the **per-chunk hash** level → upsert only new/changed chunks.
- **Dedup:** identical `chunk_hash` (even across files) embeds once.
- **Deletes/renames:** files gone from source → their chunks tombstoned; rename = delete+add, dedup
  keeps the embedding warm.
- **Append-only chat files:** re-chunk the tail; hashing means only new turns embed.
- **Resumable / crash-safe:** index in a transaction per file; never leave a half-written doc.
- **Backend down:** Ollama unreachable → queue + exponential backoff; `qmx index` is idempotent and
  resumable.
- **Concurrency:** SQLite WAL, single writer, many readers (MCP query never blocks indexing).
- **Excludes:** `.git`, `node_modules`, `dist/`, lockfiles, binaries, size cap; logged, not silently
  skipped.

## Interaction surfaces — the MCP server is primary

qmx runs as a **resident daemon** that owns the index + the background loops (watch / index /
consolidate). It has three client faces, all talking to the one daemon:

```
              ┌───────────────── qmx daemon (resident) ─────────────────┐
              │  owns the index + runs watch / index / consolidate loops │
              └───────┬───────────────┬───────────────────┬─────────────┘
        MCP interface │      CLI       │      capture      │
      (agents / CC)   │   (you)        │   (Stop hook)     │
   mcp__qmx__search…  │  qmx query …   │  qmx capture      │
```

Why a daemon (not a bare stdio MCP server): a stdio server dies when Claude Code exits, which would
kill the "always-on" capture/consolidation. The daemon stays up; MCP/CLI/hook are its clients.

### MCP tools (the main way Claude Code interacts with qmx)

| Tool | Capability | Returns |
|---|---|---|
| `search_code(query, k)` | #1 code | ranked code chunks with `file:line` |
| `recall(query, k)` | #2 memory | matching past chat turns + expand-to-conclusion |
| `lessons(query \| topic, type?, k)` | #3 learnings | relevant decisions / mistakes / howtos / preferences |
| `query(text, kind?, k)` | unified | searches the whole flat KB (`kind` optional filter) |
| `get(id)` / `expand(id)` | all | full source / surrounding turns |
| `status()` | ops | index stats, last index time, Ollama health |
| `add_learning(...)` / `consolidate()` *(optional write)* | #3 | explicitly save a lesson / trigger distillation |

Tools appear in Claude Code as `mcp__qmx__*`.

### CLI (human surface)

`qmx serve` (start daemon), `qmx index <path...>`, `qmx backfill-chats`, `qmx query "..."`,
`qmx lessons <topic>`, `qmx status`, `qmx capture` (hook entrypoint).

### Claude Code wiring

- **Register** the qmx MCP endpoint in settings (`mcpServers`) → tools become `mcp__qmx__*`.
- **Read door** = MCP (`search_code` / `recall` / `lessons`). **Write door** = the `Stop` hook
  (`qmx capture`).
- **Proactive injection (the payoff):** a `SessionStart` hook calls `qmx lessons` for the current
  context and injects relevant learnings — so the agent *starts* already knowing "last time
  bucket-level IAM failed; use project-level," without being asked. This is what makes #3 actually
  "learn from mistakes" rather than being a passive lookup.
- **Transport:** the daemon serves an HTTP MCP endpoint (Claude Code connects directly, or via a thin
  stdio bridge) so the resident background loops survive independently of any Claude Code session.

## Phasing

| Phase | Deliverable | Acceptance |
|---|---|---|
| **0** | `store.py` schema + migrations; `config.py`; `embed.py` Ollama client | round-trip: embed 3 strings, store, cosine top-k returns them |
| **1** | Code vertical slice: `chunk/code.py` + `index.py` + `search.py` + `qmx query` | index a local repo; a known function is returned in top-5 for a by-meaning query |
| **2** | **Robustness core**: incremental reindex, dedup, tombstones, `watch.py` | edit 1 file → only its chunks re-embed; delete file → chunks gone; unchanged run = ~0 embeds |
| **3** | Resident `mcp_server.py` + Claude Code wiring (rerank via llama.cpp `HttpReranker`, off by default — see [ml-notes TD-1](./qmx-ml-notes.md)) | MCP `query` callable from Claude Code |
| **4** | **Chats**: `chunk/chat.py`, `qmx backfill-chats`, Stop-hook `capture.py` | 86 transcripts searchable; a new turn is queryable within seconds |
| **5** | Hardening: backend-down, concurrency, huge files, retries + benchmarks | kill Ollama mid-index → resumes cleanly; index+query concurrently; perf numbers recorded |

## Open questions (decide as we hit them)

1. Chat capture: index **raw** turns, **summarized**, or both? (plan: raw now, distilled later)
2. ~~Index location~~ **Decided:** single `~/.qmx/index.db` **on the DGX Spark** (flat KB, one DB,
   GPU-adjacent) — see [qmx-deployment.md](./qmx-deployment.md).
3. Exact Qwen sizes (0.6B vs 4B/8B embed; reranker size) — pick in Phase 0 by speed/quality **on the
   Spark (GB10)**, the prod target.
4. Relationship to `~/.claude/.../memory/`: keep curated layer + qmx as full-recall — assumed yes

> **Decided:** no projects / collections / scoping — qmx is one flat knowledge base; queries search
> everything, with `kind` as an optional routing filter only. All chats from `~/.claude/projects/*`
> are indexed into the one KB.

## References

- qmd (origin, MIT): https://github.com/tobi/qmd
- sqlite-vec: https://github.com/asg017/sqlite-vec
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Qwen3 Embedding / Reranker: https://github.com/QwenLM
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
