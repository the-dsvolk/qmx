# qmx — Infrastructure Runbook

How the running qmx deployment is wired: the **Ollama backend** on the DGX Spark (installed,
persisted across reboots, and bound for LAN access) and the **qmx servers** (a resident one on the
Spark, a local one on the Mac). This is the concrete "what's actually running"; the design rationale
is in [`plan/qmx-deployment.md`](./plan/qmx-deployment.md).

Everything on the Spark is installed **rootless** — it has no passwordless `sudo`, so no `apt`, no
system services; user-space installs + `systemd --user` only.

## Topology / ports

| Host | Service | Bind | Purpose |
|---|---|---|---|
| Spark `spark-0e81.local` | Ollama | `0.0.0.0:11434` | Qwen embeddings (GPU); reachable on the LAN |
| Spark | qmx MCP (resident) | `0.0.0.0:8765` | shared index served to any client |
| Mac | qmx MCP (local) | `127.0.0.1:8765` | personal index (Architecture B) |

LAN-only trust model: the `0.0.0.0` binds expose Ollama and the Spark MCP to the local network
(mDNS `*.local`). Fine on a trusted LAN — do **not** do this on an untrusted network without a
firewall/reverse proxy.

---

## Spark — one-time installs (rootless)

- **uv** → `~/.local/bin/uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Ollama** → `~/.local/ollama/` from the GitHub release **arm64** asset (the `ollama.com/download`
  `.tgz` URLs 404; assets are now `.tar.zst`):
  ```bash
  curl -fSL https://github.com/ollama/ollama/releases/download/v0.32.1/ollama-linux-arm64.tar.zst -o /tmp/o.tar.zst
  mkdir -p ~/.local/ollama && tar --use-compress-program=unzstd -xf /tmp/o.tar.zst -C ~/.local/ollama
  ```
  Binary: `~/.local/ollama/bin/ollama`.
- **GPU note (GB10):** the GB10 is CUDA compute **12.1 (sm_121)**. Ollama's bundled `cuda_v12`
  runner *skips* it ("compute capability not in compiled architectures") but **`cuda_v13` supports
  it** and Ollama 0.32.1 auto-selects it (≈118 GiB VRAM available). GPU inference works.
- **Model:** `ollama pull qwen3-embedding:0.6b` — output **dim 1024** (matches qmx's `embed_dim`).

## Spark — persist Ollama across reboots + bind for remote access

Ollama runs as a **`systemd --user`** service, **bound to `0.0.0.0`** so the Mac can embed against
it over the LAN (it was originally `127.0.0.1`, unreachable off-box).

`~/.config/systemd/user/ollama.service`:

```ini
[Unit]
Description=Ollama (Qwen models for qmx)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=OLLAMA_HOST=0.0.0.0:11434          # 0.0.0.0 -> reachable from the Mac; localhost still works
Environment=OLLAMA_KEEP_ALIVE=-1               # never evict models between turns
Environment=OLLAMA_NUMA=false                  # NUMA-aware loading disabled (GB10 unified memory)
Environment=OLLAMA_MAX_LOADED_MODELS=1         # single model to reduce contention
Environment=OLLAMA_FLASH_ATTENTION=1           # enable flash attention for throughput
Environment=OLLAMA_KV_CACHE_TYPE=q8_0          # higher-precision KV cache (q8_0 vs default f16)
Environment=OLLAMA_NUM_THREADS=10              # CPU thread count for inference

ExecStart=%h/.local/ollama/bin/ollama serve
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

Enable it, and enable **linger** so the user manager (and thus the service) starts at boot without a
login — `enable-linger` works without sudo here:

```bash
loginctl enable-linger                       # user manager starts at boot; survives logout/reboot
export XDG_RUNTIME_DIR=/run/user/$(id -u)    # needed for `systemctl --user` over SSH
systemctl --user daemon-reload
systemctl --user enable --now ollama.service
curl -fsS http://localhost:11434/api/version # from the Spark
# from the Mac: curl http://spark-0e81.local:11434/api/version
```

**Reboot survival = linger `yes` + unit `enabled` + `WantedBy=default.target`.** `Restart=on-failure`
covers crashes.

## Spark — resident qmx MCP server (shared index)

Serves a shared index to any client on the LAN. Runs from the checked-out repo via `uv run`.

`~/.config/qmx/env`:

```ini
QMX_OLLAMA_URL=http://localhost:11434
QMX_EMBED_MODEL=qwen3-embedding:0.6b
QMX_EMBED_DIM=1024
QMX_DB_PATH=/home/dsvolk/.qmx/index.db
QMX_MCP_HOST=0.0.0.0
QMX_MCP_PORT=8765
```

`~/.config/systemd/user/qmx-mcp.service`:

```ini
[Unit]
Description=qmx MCP server (resident)
After=ollama.service network-online.target
Wants=ollama.service

[Service]
Type=simple
WorkingDirectory=%h/GitHub/the-dsvolk/qmx
EnvironmentFile=%h/.config/qmx/env
ExecStart=%h/.local/bin/uv run --project %h/GitHub/the-dsvolk/qmx qmx serve --transport http
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now qmx-mcp.service
# MCP endpoint: http://spark-0e81.local:8765/mcp
```

The clone at `~/GitHub/the-dsvolk/qmx` tracks `main` (`git pull` + `uv sync` + `systemctl --user
restart qmx-mcp` to update). Index built with `qmx index <path>` into `~/.qmx/index.db`.

## Spark — reranker server (`qmx-rerank.service`, GPU)

A cross-encoder reranker: **llama.cpp `llama-server --reranking`** serving **Qwen3-Reranker-0.6B**
on the GB10. Built from source (no prebuilt arm64/Blackwell image exists):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDACXX=/usr/local/cuda-13.0/bin/nvcc
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF
cmake --build build --config Release -j
uvx --from "huggingface_hub[cli]" hf download ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF \
  qwen3-reranker-0.6b-q8_0.gguf --local-dir ~/models
```

`~/.config/systemd/user/qmx-rerank.service` → `llama-server --model ~/models/qwen3-reranker-0.6b-q8_0.gguf
--reranking --host 0.0.0.0 --port 8081 -ngl 99 -b 8192 -ub 8192`
(`Environment=LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:%h/llama.cpp/build/bin`).
`-b/-ub 8192` is required — the default 512 rejects long chunks (HTTP 500). Endpoint:
`http://spark-0e81.local:8081/v1/rerank`. Clients enable it with `rerank_url` (off by default).

## Spark — learnings/consolidation model (chat, GPU)

The learnings tier (`qmx consolidate` / `session-end`) distils chats into lessons with a **Qwen chat
model** — the one qmx component where model quality matters (judgment: what's a durable lesson, is
this a dup or a supersede?). It runs on the **same Ollama service** as embeddings (no new unit) —
just pull the model:

```bash
export OLLAMA_HOST=127.0.0.1:11434
~/.local/ollama/bin/ollama pull qwen3.6:35b-a3b   # MoE 35B/3B-active; ~23 GB, fits the ~118 GB VRAM
~/.local/ollama/bin/ollama list                   # confirm it appears alongside qwen3-embedding:0.6b
```

> **Verified working (2026-07-18).** Pulled on the Spark (`qwen3.6:35b-a3b`, 23 GB, MoE) and served by
> the shared Ollama. Verified **end-to-end from the client**: on the Mac, `Settings.load()` →
> `OllamaChat` → the production `extract_learnings` path (`think=false` + schema-constrained `format`)
> reaches the Spark over `QMX_OLLAMA_URL` and returns a valid `{"learnings":[…]}` extraction with
> `type` correctly on-enum (`decision|mistake|howto`). Warm round-trip ≈10 s; first (cold) call ≈44 s
> incl. loading the model into VRAM.

- **Where it's configured — on the client, not here.** Consolidation runs where the CLI/hooks run
  (the Mac), talking to *this* Ollama over `QMX_OLLAMA_URL`; the Spark's resident MCP server never
  calls the chat model (it only serves retrieval). So the Spark just needs the model **pulled**; the
  model *name* is set on the Mac in **`~/.qmx/config.toml`** (`chat_model = "qwen3.6:35b-a3b"`, or
  `QMX_CHAT_MODEL`) — see the Mac section below. Default is `qwen3.6:35b-a3b`, never hardcoded; to
  swap models, pull the new tag on the Spark and change that one value.
- **Batch, low-QPS.** Consolidation is a few calls at session end, so throughput is irrelevant;
  `session-end` runs it **detached** so it never blocks a session closing. Nothing is resident.
- **Deferred upgrade (only if v1 lessons are weak):** a `Qwen3.5-122B-A10B` in **NVFP4** on a
  **vLLM** server (NVFP4 ≠ GGUF → not Ollama). Point `chat_model`/`QMX_OLLAMA_URL` at it if built.
  See [`plan/qmx-learnings.md`](./plan/qmx-learnings.md) (*Model decision*).

Client-side hooks that call this model (`SessionStart` inject, `SessionEnd` consolidate) are wired in
Claude Code `settings.json` on the Mac — see the **Learnings** section of [`README.md`](./README.md).

## Mac — local qmx (Architecture B: index local, embed on the Spark)

- Install: `uv tool install "git+https://github.com/the-dsvolk/qmx"` → `~/.local/bin/qmx`.
- Config `~/.qmx/config.toml`: `ollama_url = "http://spark-0e81.local:11434"`,
  `embed_model = "qwen3-embedding:0.6b"`, `embed_dim = 1024`, `mcp_host = "127.0.0.1"`,
  `mcp_port = 8765`. Index at `~/.qmx/index.db`.
  - **Learnings model:** `chat_model = "qwen3.6:35b-a3b"` (the consolidation judge — this is the one
    the `qmx consolidate` / `session-end` hook uses against the Spark's Ollama; must be pulled there).
    It defaults to this value, so the line is optional unless you swap models.
  - **Reranker (optional):** `rerank_url = "http://spark-0e81.local:8081"` enables the cross-encoder.
- Claude Code: `claude mcp add --transport http --scope user qmx http://127.0.0.1:8765/mcp`.

**launchd agents live in `~/Library/LaunchAgents/`** — they are **client-side machine state, not in
the repo** (same as the Spark's systemd units above; the plists are reproduced here in full).
Install each with `launchctl load -w ~/Library/LaunchAgents/<name>.plist`. Three exist:

`com.qmx.serve.plist` — the always-on MCP server (`RunAtLoad`+`KeepAlive`; log `~/.qmx/serve.log`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.qmx.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YZ0315/.local/bin/qmx</string>
    <string>serve</string><string>--transport</string><string>http</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/YZ0315/.qmx/serve.log</string>
  <key>StandardErrorPath</key><string>/Users/YZ0315/.qmx/serve.log</string>
</dict>
</plist>
```

`com.qmx.watch.plist` — keeps `code_roots` reindexed on save; identical shape but
`ProgramArguments = [qmx, -v, watch]` and log `~/.qmx/watch.log`.
(`com.qmx.consolidate.plist` — the learnings sweep — is shown in the next section.)

Full step-by-step is in [`QUICKSTART.md`](./QUICKSTART.md).

## Mac — learnings triggers (what runs consolidation "constantly")

Consolidation runs on the **client** (transcripts + index are here; it calls the Spark only for the
model). Two mechanisms, installed on the Mac — **no daemon, event-driven + a nightly safety net:**

1. **Claude Code hooks** (`~/.claude/settings.json`) — the primary, per-session trigger:
   ```json
   "SessionStart": [{ "matcher": "startup", "hooks": [{ "type": "command", "command": "/Users/YZ0315/.local/bin/qmx session-start" }] }],
   "SessionEnd":   [{ "hooks": [{ "type": "command", "command": "/Users/YZ0315/.local/bin/qmx session-end" }] }]
   ```
   `session-end` spawns `qmx consolidate` **detached** (never blocks session close); `session-start`
   injects scope-matched lessons. Both are best-effort (exit 0). Alongside the existing `Stop →
   qmx capture` hook.

   **Cursor** (`~/.cursor/hooks.json`) is a second trigger source into the same store — `stop →
   qmx capture --source cursor` and `sessionEnd → qmx session-end` (detached consolidate). Cursor
   payloads have no `cwd` and don't reliably carry `transcript_path` on stdin, so the hook scripts
   merge `CURSOR_TRANSCRIPT_PATH` / `CURSOR_PROJECT_DIR` (env) into the payload; scope resolves from
   `workspace_roots[0]` / `CURSOR_PROJECT_DIR`. **No `sessionStart`** is wired (no injection at
   Cursor start). See `QUICKSTART.md` §7 for the config; the `--source` flag must exist in the
   installed binary first. Cloud agents don't fire `sessionStart`/`sessionEnd` (local-only).
2. **Daily sweep** — a launchd catch-all for sessions the hook missed (crashes, backfilled/old
   transcripts). Runs `qmx consolidate --all` at **03:00 daily**; cheap + idempotent (the
   `consolidated` cursor means it only distils *new* turns). `~/Library/LaunchAgents/com.qmx.consolidate.plist`:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>Label</key><string>com.qmx.consolidate</string>
     <key>ProgramArguments</key>
     <array>
       <string>/Users/YZ0315/.local/bin/qmx</string><string>consolidate</string><string>--all</string>
     </array>
     <key>StartCalendarInterval</key>
     <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
     <key>RunAtLoad</key><false/>
     <key>StandardOutPath</key><string>/Users/YZ0315/.qmx/consolidate.log</string>
     <key>StandardErrorPath</key><string>/Users/YZ0315/.qmx/consolidate.log</string>
   </dict>
   </plist>
   ```

   (`StartCalendarInterval` = fixed clock time; launchd runs it on next wake if the Mac was asleep at
   03:00 — unlike cron, which would skip it. Swap for `StartInterval` `<integer>21600</integer>` to run
   every 6 h instead.)
   > Caveat: `--all` has no per-transcript cwd, so swept lessons are **global** (`scope=NULL`); the
   > per-session `SessionEnd` path derives the repo scope from `cwd` and scopes correctly.

Manage: `launchctl load -w|unload ~/Library/LaunchAgents/com.qmx.consolidate.plist`;
`launchctl start com.qmx.consolidate` to run the sweep now; `tail -f ~/.qmx/consolidate.log`.
Requires the installed `qmx` to include the learnings commands (`uv tool upgrade qmx` once the
learnings PR is on `main`). **Verified:** consolidating a real session produced 7 scoped lessons in
~32 s, queryable via `qmx lessons`.

---

## Managing it

**Spark (`systemd --user`)** — always `export XDG_RUNTIME_DIR=/run/user/$(id -u)` first over SSH:

```bash
systemctl --user status ollama qmx-mcp
systemctl --user restart ollama            # or qmx-mcp
journalctl --user -u qmx-mcp -f            # live request log (POST /mcp per tool call)
journalctl --user -u ollama -n 50
loginctl show-user "$USER" -p Linger       # expect Linger=yes
```

**Mac (launchd)** — three agents: `com.qmx.serve` (MCP), `com.qmx.watch` (code reindex),
`com.qmx.consolidate` (daily learnings sweep, 03:00):

```bash
launchctl list | grep qmx                                     # serve, watch, consolidate
# swap com.qmx.serve for .watch / .consolidate as needed:
launchctl unload ~/Library/LaunchAgents/com.qmx.serve.plist   # stop
launchctl load  -w ~/Library/LaunchAgents/com.qmx.serve.plist # start / enable at login
tail -f ~/.qmx/serve.log

# learnings daily sweep (com.qmx.consolidate):
launchctl start com.qmx.consolidate                           # run the --all sweep now
tail -f ~/.qmx/consolidate.log
```

## Rebuilding from scratch

The index is a **rebuildable shadow** — safe to delete. If the DB is corrupt or the embedding
model/dim changes: `rm ~/.qmx/index.db*` then re-`qmx index`. Nothing else needs re-doing; the
services and config above are the only durable state.
