# qmx — Quickstart

Get qmx searching your code by meaning, from the CLI and from Claude Code, in a few minutes.

qmx splits into two jobs: **embeddings** (GPU-heavy, done by Ollama) and the **index + search**
(a local SQLite file). They don't have to live on the same machine — that's the `QMX_OLLAMA_URL`
seam. The recommended personal setup is:

> **Index + serve on your laptop, embed on the GPU box (the DGX Spark).**
> Your code and index stay local; only embedding calls go to the Spark. No files to sync.

If instead you want one shared index served to many clients, run everything on the Spark — see the
"Server-resident" note at the bottom and [`plan/qmx-deployment.md`](./plan/qmx-deployment.md).

For how the backend (Ollama on the Spark) is set up and kept alive, see [`INFRA.md`](./INFRA.md).

---

## 1. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed.
- An **Ollama backend** reachable over the network, with the Qwen embedding model pulled. In this
  setup that's the DGX Spark at `http://spark-0e81.local:11434` serving `qwen3-embedding:0.6b`
  (see [`INFRA.md`](./INFRA.md) to (re)create it).

## 2. Install the CLI

```bash
uv tool install "git+https://github.com/the-dsvolk/qmx"   # install
uv tool upgrade qmx                                        # update to the latest main later
qmx --help
```

`uv tool upgrade qmx` pulls the newest `main` and rebuilds the isolated tool env — run it after new
features land (e.g. chat memory) to refresh the `qmx` command.

## 3. Configure (point at the embedding backend)

qmx reads `~/.qmx/config.toml` automatically. Create it:

```toml
# ~/.qmx/config.toml
ollama_url  = "http://spark-0e81.local:11434"   # GPU box does the embeddings
embed_model = "qwen3-embedding:0.6b"
embed_dim   = 1024                              # must match the model
mcp_host    = "127.0.0.1"                       # local-only MCP server
mcp_port    = 8765
# db_path defaults to ~/.qmx/index.db

# Optional cross-encoder rerank (sharpens top-k). Off if unset. Points at a llama.cpp
# `llama-server --reranking` (Qwen3-Reranker) — see INFRA.md. Fails soft to RRF if down.
# rerank_url = "http://spark-0e81.local:8081"
```

Any field can be overridden per-command with a `QMX_*` env var (e.g. `QMX_OLLAMA_URL`,
`QMX_EMBED_MODEL`, `QMX_DB_PATH`).

Sanity check:

```bash
qmx status        # shows resolved config + (empty) index stats
```

### Choosing the embedding model

`embed_model` is **any embedding model your Ollama backend serves**, and `embed_dim` **must match
its output width**. To switch models:

1. **Pull it on the backend** (the machine `ollama_url` points at):
   ```bash
   ollama pull qwen3-embedding:0.6b     # run on the Ollama host, or: ollama pull <other-model>
   ```
2. **Find its dimension** — embed a probe string and count the vector:
   ```bash
   curl -s "$OLLAMA_URL/api/embed" -d '{"model":"qwen3-embedding:0.6b","input":["x"]}' \
     | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"
   ```
3. **Set `embed_model` + `embed_dim`** in `config.toml` to match what step 2 printed.

Guidance:
- **Bigger model = better recall, but slower and more VRAM.** The Qwen3-Embedding family comes in
  0.6B / 4B / 8B; **`qwen3-embedding:0.6b` → 1024 dims** (a good default). Larger variants emit wider
  vectors — probe to confirm. Any Ollama embedding model works (e.g. `nomic-embed-text`,
  `mxbai-embed-large`, `bge-m3`) as long as `embed_dim` matches.
- **The model and dim are baked into the index.** qmx records them and **refuses to open an index
  built with a different model/dim** (it tells you to rebuild). So changing the embedding model means
  re-embedding: `rm` the DB (or use a fresh `QMX_DB_PATH`) and re-`qmx index`.
- **Use the same model for indexing and querying** — mismatched vector spaces return garbage.

## 4. Index a repo and search it

```bash
qmx index ~/code/my-project        # walks code (kind=code) + markdown (.md, kind=doc), writes ~/.qmx/index.db
qmx query "where do we retry failed requests" -k 5
qmx query "egress quota" --kind doc         # search only docs (e.g. a repo's kb/*.md)
```

Code is chunked AST-aware (tree-sitter); `.md` docs are chunked by headings and stored as
`kind=doc`, so a repo's `kb/`, READMEs, and design notes are searchable too (unified `query` returns
both; `--kind code`/`--kind doc` filters).

Re-running `qmx index` is incremental — only changed chunks re-embed, deleted files are dropped.

## 5. Use it from Claude Code

Run the resident server (always-on) and register it once.

**Always-on via launchd (macOS):** create `~/Library/LaunchAgents/com.qmx.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.qmx.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>REPLACE_HOME/.local/bin/qmx</string>
    <string>serve</string><string>--transport</string><string>http</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>REPLACE_HOME/.qmx/serve.log</string>
  <key>StandardErrorPath</key><string>REPLACE_HOME/.qmx/serve.log</string>
</dict></plist>
```

```bash
launchctl load -w ~/Library/LaunchAgents/com.qmx.serve.plist
# (or just run it in a terminal: qmx serve)
```

**Register with Claude Code** (user scope = all projects):

```bash
claude mcp add --transport http --scope user qmx http://127.0.0.1:8765/mcp
claude mcp get qmx      # → ✔ Connected
```

Open a **new** Claude Code session (tools load at startup) and the following appear:
`mcp__qmx__query`, `mcp__qmx__search_code`, `mcp__qmx__recall`, `mcp__qmx__get`, `mcp__qmx__status`.
Ask something like *"use qmx to find the rate-limiter"* (`recall` searches past chats specifically;
`query` searches everything).

> Tools are **available** to the agent, not auto-run: Claude calls them when relevant or when you
> ask. Proactive "always consult qmx" wiring (a hook) is future work.

## 6. Keep it live, manage projects & maintenance

```bash
qmx watch ~/code/my-project   # reindex on save (create/modify/delete)
qmx watch                     # no args → watch everything in code_roots
qmx index ~/code/my-project   # re-index on demand (incremental; --force to re-embed all)

qmx sources                   # list what's indexed (grouped by repo + counts)
qmx remove ~/code/my-project  # drop a file or whole directory subtree from the index
qmx gc                        # purge tombstoned (removed/edited-away) chunks to reclaim space

qmx status                    # documents / chunks / mentions / live vs tombstoned
```

**Managing a project's lifecycle:**
- **Re-index** — just run `qmx index <path>` again; it's incremental (unchanged files skip, changed
  chunks re-embed, files deleted within the tree are pruned). `qmx watch` does this continuously.
- **Delete a project** — `qmx remove <path>` drops it from the index (then `qmx gc` reclaims). Or, if
  you gave the project its own DB (`QMX_DB_PATH=~/.qmx/<name>.db`), just `rm ~/.qmx/<name>.db*`.
- **Multiple codebases** — index several into one DB (they're distinguished by `path`), or keep one
  DB per project and switch with `QMX_DB_PATH`.
- **Keep it all in sync with one command** — list the repos you want indexed in `code_roots`, then
  `qmx refresh` re-indexes those **plus** chats and memory into the flat KB in a single pass:
  ```toml
  # ~/.qmx/config.toml
  code_roots = ["~/GitHub/Cruise/xtorch", "~/GitHub/Cruise/cpe-intelligence-main"]
  ```
  ```bash
  qmx refresh          # index code_roots + backfill-chats + index-memory (incremental)
  ```

## 7. Chat memory — index past chats & capture new ones

qmx also indexes your Claude Code conversation history (as `kind=chat`, into the **same** flat KB as
code), so you can recall past sessions by meaning.

**Backfill existing transcripts** — one-time import of `~/.claude/projects/*/*.jsonl`:

```bash
qmx backfill-chats                              # all transcripts in ~/.claude/projects
qmx backfill-chats --projects /path/to/projects # a custom location
qmx query "when did we discuss retry backoff" --kind chat
```

It parses the JSONL directly and keeps only human/assistant **text** — `thinking`, tool calls/results,
side-chains, and system reminders are dropped. Re-running is cheap: only new turns embed.

**Capture new turns live** — add a Claude Code `Stop` hook so every finished turn is indexed. In
`~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "qmx capture" } ] }
    ]
  }
}
```

Claude Code pipes the turn's transcript path to `qmx capture` on stdin; it incrementally indexes just
the new turn(s). It's **best-effort and never blocks a turn** (any failure is swallowed). Then
`mcp__qmx__recall` (or `qmx query --kind chat`) surfaces those conversations.

### Claude memory files

qmx also indexes your **curated Claude memory** (`~/.claude/projects/*/memory/*.md` — the `MEMORY.md`
index + per-fact notes) as `kind="memory"`, so those facts are searchable too. `qmx capture` refreshes
the current project's memory automatically (the `memory/` sibling of the transcript); to backfill all
of it:

```bash
qmx index-memory               # sweeps the configured memory roots
qmx query "spark ssh" --kind memory
```

The roots are a config list — set `memory_globs` in `config.toml` (or `QMX_MEMORY_GLOBS=a,b`) to add
other locations (dirs are scanned for `*.md`; a `.md` path is taken directly):

```toml
memory_globs = ["~/.claude/projects/*/memory", "~/.claude/CLAUDE.md"]
```

## Verifying a query actually hit qmx

Tail the server log while you ask a question:

```bash
tail -f ~/.qmx/serve.log      # a POST /mcp line appears when a tool fires
```

In the Claude Code transcript you'll also see an `mcp__qmx__query` tool block.

---

## Alternative: server-resident (one shared index)

Instead of per-laptop indexes, run qmx **on the Spark** (index + `qmx serve` bound to the LAN) and
point every client's Claude Code at `http://spark-0e81.local:8765/mcp`. Good when you want a single
shared index or always-on chat capture. Setup and the systemd units are in [`INFRA.md`](./INFRA.md).
