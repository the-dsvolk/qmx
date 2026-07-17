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
uv tool install "git+https://github.com/the-dsvolk/qmx"
qmx --help
```

Update later with `uv tool upgrade qmx`.

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
```

Any field can be overridden per-command with a `QMX_*` env var (e.g. `QMX_OLLAMA_URL`).

Sanity check:

```bash
qmx status        # shows resolved config + (empty) index stats
```

## 4. Index a repo and search it

```bash
qmx index ~/code/my-project        # walks code files, embeds via the Spark, writes ~/.qmx/index.db
qmx query "where do we retry failed requests" -k 5
```

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
`mcp__qmx__query`, `mcp__qmx__search_code`, `mcp__qmx__get`, `mcp__qmx__status`. Ask something like
*"use qmx to find the rate-limiter"*.

> Tools are **available** to the agent, not auto-run: Claude calls them when relevant or when you
> ask. Proactive "always consult qmx" wiring (a hook) is future work.

## 6. Keep it live, manage projects & maintenance

```bash
qmx watch ~/code/my-project   # reindex on save (create/modify/delete)
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
