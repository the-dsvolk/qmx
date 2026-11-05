# qmx — Infrastructure Runbook

How the running qmx deployment is wired: the **model backends** on the DGX Spark — **Ollama** for
embeddings (installed, persisted across reboots, and bound for LAN access), the **vLLM judge**
container and the llama.cpp **reranker** — and the **qmx servers** (the primary local one on the
Mac, plus an alternative resident one on the Spark). This is the concrete "what's actually
running"; the design rationale is in [`plan/qmx-deployment.md`](./plan/qmx-deployment.md).

Everything on the Spark is installed **rootless** — it has no passwordless `sudo`, so no `apt`, no
system services; user-space installs + `systemd --user` only.

## Topology / ports

| Host | Service | Bind | Purpose |
|---|---|---|---|
| Spark `spark-0e81.local` | Ollama | `0.0.0.0:11434` | Qwen embeddings (GPU); reachable on the LAN |
| Spark `spark-0e81.local` | vLLM (OpenAI API), Docker `qwen38-judge` | `:8000` | **production judge** `qwen3.8-judge` (`chat_url = "http://spark-0e81.local:8000/v1"`) |
| Spark | qmx MCP (resident) | `0.0.0.0:8765` | alternative deployment: shared index served to any client |
| Second spark `spark-0f50.local` | vLLM | `:8001` | judge experiments (Nemotron 3.5 Super); not production |
| Second spark `spark-0f50.local` | vLLM, Docker `qwen38-judge` (stopped, `restart=no`) | `:8000` | judge **fallback** only — see *Fallback* below |
| Mac | qmx MCP (local) | `127.0.0.1:8765` | **primary**: personal index, what Claude Code connects to (Architecture B) |

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

An alternative to the Mac-local server (which is what Claude Code uses in this deployment): serves a
shared index to any client on the LAN. Runs from the checked-out repo via `uv run`.

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

The learnings tier (`qmx consolidate` / `session-end`) distils chats into lessons with a **Qwen
judge model** — the one qmx component where model quality matters (judgment: what's a durable
lesson, is this a dup or a supersede?). It does **not** run on Ollama: since 2026-11-05 it is the
Docker container **`qwen38-judge`** on the **primary** spark `spark-0e81` — vLLM v0.33.0 serving
Qwen3.8-27B NVFP4 + DFlash2 speculative decoding as `qwen3.8-judge` on the OpenAI-compatible API
at `:8000`, with `restart=unless-stopped` and `--gpu-memory-utilization 0.4` (≈47 GB, alongside
the Ollama embedder and the reranker). A cold start takes **~8 min**. At boot the primary spark
preloads only the embedder (`ollama-warmup.service`); the judge comes back through its Docker
restart policy.

```bash
docker ps --filter name=qwen38-judge                  # on spark-0e81: is the judge up?
docker logs -f qwen38-judge                           # load progress / request log
curl -fsS http://spark-0e81.local:8000/v1/models      # from the Mac: lists qwen3.8-judge when ready
```

There is **no Ollama judge** any more: `qwen3.6:35b-a3b` (the previous judge) and
Qwen3-Coder-Next were removed from Ollama on 2026-11-05.

- **Where it's configured — on the client, not here.** Consolidation runs where the CLI/hooks run
  (the Mac), talking to the judge at `chat_url`; the Spark's resident MCP server never calls the
  judge (it only serves retrieval). The judge is set on the Mac in **`~/.qmx/config.toml`**
  (`chat_backend` / `chat_url` / `chat_model`, or `QMX_CHAT_BACKEND` / `QMX_CHAT_URL` /
  `QMX_CHAT_MODEL`) — see the Mac section below. The code defaults are the production judge; the
  consolidation code never hardcodes a model, so to swap models, serve the new one and change
  those values.
- **Batch, low-QPS.** Consolidation is a few calls at session end, so throughput is irrelevant;
  `session-end` runs it **detached** so it never blocks a session closing. qmx keeps no process
  running for it; only the vLLM container stays up.
- **Context and timeout (judge only).** vLLM's context window is fixed server-side. `chat_num_ctx`
  (default **65536**) is sent only by the `"ollama"` backend, as Ollama `num_ctx` (it was raised
  from 32768 when the retired qwen3.6 judge was judging long sessions on 26-46% of their turns).
  The judge waits up to `chat_timeout` (default **300 s**); `request_timeout` (60 s) stays short
  because it also bounds the synchronous Stop-hook embed.
- **Judge on vLLM (primary spark).** Production runs `chat_backend = "openai"`, `chat_url =
  "http://spark-0e81.local:8000/v1"`, `chat_model = "qwen3.8-judge"` (also the code defaults):
  Qwen3.8-27B NVFP4 + DFlash2 served by vLLM on the **primary** spark (thinking off,
  reasoning-first JSON) — blind-graded 6.3 vs 5.1/10 against qwen3.6 on 12 sessions (2026-11-03),
  ~2 min/session. The second spark (`spark-0f50.local`) is only for judge experiments (Nemotron 3.5
  Super on `:8001`, see *Judge filing rules v2* below), not production. Any other judge host goes
  in config. If the judge is unreachable, consolidation is **deferred**, not lost: a pass is only
  marked consolidated when it finishes, and the nightly `qmx consolidate --all` (launchd, 03:00)
  picks the turns up.
- **Fallback.** There is no Ollama judge to fall back to. The fallback is the stopped
  `qwen38-judge` container on the second spark `spark-0f50` (`restart=no`, so it does not start at
  boot). To use it, start it there and repoint the Mac:

  ```bash
  # on spark-0f50
  docker update --restart unless-stopped qwen38-judge && docker start qwen38-judge
  ```
  ```toml
  # ~/.qmx/config.toml on the Mac
  chat_url = "http://spark-0f50.local:8000/v1"
  ```

  Allow for the cold start; turns that arrive meanwhile stay queued for the next run.
- **Fixed JSON layout and unusable replies (vLLM judge).** With `response_format`, vLLM's grammar
  allows any whitespace between JSON tokens. When the judge writes an unescaped `"` in a text field,
  the string closes early and greedy decoding emits newlines until the token cap: roughly 1 in 50-100
  real calls, and on 2026-11-03 one such merge check (retried 5 x 300 s) deferred a whole `--all`.
  Every judge prompt now asks for single quotes inside values, and decisions/checks are capped at
  3072 output tokens (`chat_short_max_tokens`; still capped by `chat_max_tokens`).
  `chat_json_layout = "indent"` (opt-in; default `"schema"` = `response_format`) sends a
  per-request grammar admitting exactly `json.dumps(indent=2)`, which removes the loop entirely; it
  stays off until its decoding speed matches xgrammar's own EBNF (3-5x slower in a 2026-11-04
  check).
  An unusable reply (cut off at the cap, no reply before the timeout while `/models` still answers,
  or not JSON) fails safe for that one call: decision -> stored as new, merge/retirement check ->
  refused, dedup check -> candidate kept, extract -> its pass stays queued. Three unusable replies in
  a row, or an unreachable judge, still defer the run. Don't use vLLM's server-wide
  `disable_any_whitespace` instead: it forces compact JSON and corrupted fields
  (`statement: ',detail'`). Evidence: `~/knowledge/qwen-judge-eval/runaway-debug/`.
- **Long sessions & one-off gate.** Sessions are read in passes of ≤200 turns / ≤100K characters, each
  marked when done (previously only the first 200 turns were ever read). `consolidate_min_turns` (8)
  and `consolidate_min_chars` (4000) skip one-off sessions, leaving them to accumulate.
- **Merge & retirement guardrails.** An `update` must quote, verbatim, where the old lesson's core
  claim survives (checked in code); a `deprecate` or `supersede` must quote the part of the new
  lesson that contradicts the old one. Otherwise (an unusable reply included) the old lesson stays
  intact: a refused `update`/`deprecate` stores the candidate as new; a refused `supersede` sends it
  through the near-duplicate check, which drops or merges it if it restates a live lesson and
  otherwise stores it as new in its own words. `qmx consolidate` reports blocked supersedes.
  Supersede was unguarded until a 2026-11-04 end-to-end eval caught it retiring three lessons the
  sessions supported (#1852, #380, #1142). Replayed through the check, the production judge blocks
  #380 but still "verifies" #1852 and #1142 with a real but non-contradicting quote, so the shared
  prompt is a first line, not a complete fix; the stricter check is part of filing rules v2 below.
- **Reasoning-first JSON.** All five judge schemas start with a required `reasoning` string, so the
  model analyses before it decides (thinking stays off). Eval 2026-11-02, 12 sessions blind-graded:
  precision 3.8 vs 3.3/5, +46% judge time, no call over 60 s; Ollama thinking mode was ~10× slower.
  Essential for Qwen3.8 with thinking off (without it, 2-3 of 11 sessions returned empty lesson
  lists). `judge_reasoning_field = false` removes the field and its wording from all five calls,
  for a model that thinks natively.
- **Judge filing rules v2 (opt-in: `judge_filing_rules = "v2"`; default `"current"`).** In the 12
  production merges after the Qwen3.8 cutover (2026-11-04/05), 6 did not keep the old lesson's first
  sentence (5 rewritten, 1 demoted to detail; mostly benign rewordings, but nothing enforced it); a
  Super run replaced three headlines outright and its own merge check passed all three. v2 changes
  four things, each active only with `"v2"`:
  1. **Main rule = first sentence of the statement, computed in code** (`main_rule()`: a leading
     `[type]` tag stripped, "e.g."/"vs."/"incl." do not end a sentence) and handed to every prompt
     and check, instead of letting the judge pick a "core claim". Decision and near-duplicate
     prompts compare main rules, not topics, with an operational same-rule test ("someone who
     follows the old main rule would do or know the same thing") and "same main rule → `update` or
     `drop`, not `new`" against over-caution.
  2. **Headline backstop** (`headline_kept()`), decided in code before the LLM merge check, on both
     merge paths: the merged statement must begin with the old first sentence (case, whitespace,
     quotes, backticks, `[type]` tags and dash style ignored; ends on a word boundary). A pure
     extension or a corrected number passes. A refused decision `update` goes through the
     near-duplicate check once (whose merge passes the backstop too); a refused near-duplicate merge
     is dropped. On the eval merges it caught every known clobber and fired 0 times under the v2
     prompts; under the current prompts it would block about half of Qwen's acceptable rewordings,
     which is why it ships only with v2.
  3. **Off-rule check in the merge check.** The judge lists clauses of the merged text that state a
     different rule than OLD MAIN RULE (the topic-match merge: recommender facts appended to an
     unrelated CUD lesson). By default such a merge is applied, logged as a warning and counted
     (`offrule_flagged`); with `judge_offrule_block = true` it is refused and the candidate stored as
     new (never dropped), on the update and the near-duplicate path. The block was the default
     until the PR #39 validation (2026-11-05) showed it refusing merges that kept the headline and
     should have gone through (they became near-duplicate lessons). Lost facts and unsupported
     additions are logged only.
  4. **Two-quote retirement check.** `supersede`/`deprecate` need the candidate's contradicting
     words *and* the words of OLD MAIN RULE that become false, both verified verbatim in code; the
     old-side quote must be at least 3 words (or the whole main rule if shorter), because a one-word
     quote ("vLLM-only") let a wrong retirement through.
  Also: no `[type]` tags or whole-value quote wrapping in judge-written text (stripped in code),
  and `qmx consolidate` reports `headline_blocked` / `offrule_blocked` / `offrule_flagged`. The
  decision prompt no longer tells the judge to choose `new` "when you cannot tell": in validation
  Qwen3.8 filed same-rule candidates as new lessons. With v2 on Qwen3.8, set
  `chat_short_max_tokens = 6144`: at 3072 the reasoning-first merge check was cut off on long
  lessons (an unusable reply refuses the merge). The precursor of these
  prompts ("rules-v2") replayed the 15 known filing failures cleanly (Qwen3.8 15/15 vs 11/15 with
  the current prompts; Super 15/15), but Qwen then made 0 merges; the same-rule test above targets
  that over-caution. The final texts are not validated yet: v2 becomes the default only after the
  end-to-end validation (Qwen non-regression) in
  `~/knowledge/qwen-judge-eval/prompt-research/PROPOSAL.md` §C.1 passes.
- **Judge client settings (all off by default).** With defaults the request body is byte-identical
  to the validated Qwen3.8 request (tests pin it), so a different judge model is config only:

  | Setting (env) | Default | Effect |
  |---|---|---|
  | `chat_temperature` (`QMX_CHAT_TEMPERATURE`) | `0.0` | sampling temperature (vLLM and Ollama) |
  | `chat_top_p` (`QMX_CHAT_TOP_P`) | `0.0` = not sent | `top_p`, sent only when > 0 |
  | `chat_template_kwargs` (`QMX_CHAT_TEMPLATE_KWARGS`) | `""` | JSON object (TOML inline table or JSON string) merged over the `enable_thinking`/`reasoning_effort` that `chat_think` computes; the override wins. vLLM only. Invalid JSON fails at startup |
  | `chat_short_max_tokens` (`QMX_CHAT_SHORT_MAX_TOKENS`) | `3072` | cap of decision / merge / retirement / near-duplicate calls (thinking shares it) |
  | `judge_reasoning_field` (`QMX_JUDGE_REASONING_FIELD`) | `true` | reasoning-first field in all five calls (true/false, 1/0, yes/no, on/off) |
  | `judge_filing_rules` (`QMX_JUDGE_FILING_RULES`) | `"current"` | `"current"` or `"v2"` |
  | `judge_offrule_block` (`QMX_JUDGE_OFFRULE_BLOCK`) | `false` | v2 only: refuse (true) or log (false) a merge that adds a different rule |

  **Nemotron 3.5 Super profile** (experiments on the second spark; server unchanged: vLLM 0.32,
  Super NVFP4 + LiLiCorr, `--max-model-len 81920`, `nemotron_v3`). Native medium thinking
  replaces the reasoning field; temperature 0 with thinking gave empty sessions and runaways, full
  thinking took 25 min/session:

  ```toml
  chat_backend = "openai"
  chat_url = "http://spark-0f50.local:8001/v1"
  chat_model = "nemotron-3.5-super-vl"
  chat_think = "true"
  chat_template_kwargs = { enable_thinking = true, medium_effort = true }
  chat_temperature = 1.0
  chat_top_p = 0.95
  chat_max_tokens = 32768
  chat_short_max_tokens = 16384
  chat_timeout = 1200
  chat_json_layout = "schema"
  judge_reasoning_field = false
  judge_filing_rules = "v2"
  ```

  At ~18 tok/s a 16384-token runaway ends at the cap in ~900 s, under the 1200 s timeout; a runaway
  extract would hit the timeout first and still fail safe (its pass stays queued). Eval harnesses
  that call `consolidate_session` directly pass `judge=judge_config(settings)` (or
  `build_judge("v2", reasoning_field=False, short_max_tokens=16384)`); without `judge` it uses the
  module constants, i.e. today's judge. Under v2 the prompts differ from those constants, so a
  harness that tells call kinds apart by prompt should compare against `judge.prompts`.
- **Deferred upgrade (only if the current judge's lessons are weak):** a `Qwen3.5-122B-A10B` in
  **NVFP4** on a **vLLM** server (NVFP4 ≠ GGUF → not Ollama). Point `chat_url`/`chat_model` at it if
  built. See [`plan/qmx-learnings.md`](./plan/qmx-learnings.md) (*Model decision*).

Client-side hooks that call this model (`SessionStart` inject, `SessionEnd` consolidate) are wired in
Claude Code `settings.json` on the Mac — see the **Learnings** section of [`README.md`](./README.md).

## Mac — local qmx (Architecture B: index local, embed on the Spark) — the primary setup

- Install: `uv tool install "git+https://github.com/the-dsvolk/qmx"` → `~/.local/bin/qmx`.
- Config `~/.qmx/config.toml`: `ollama_url = "http://spark-0e81.local:11434"`,
  `embed_model = "qwen3-embedding:0.6b"`, `embed_dim = 1024`, `mcp_host = "127.0.0.1"`,
  `mcp_port = 8765`. Index at `~/.qmx/index.db`.
  - **Learnings judge:** `chat_backend = "openai"`, `chat_url = "http://spark-0e81.local:8000/v1"`,
    `chat_model = "qwen3.8-judge"` — the vLLM judge on the primary spark that the
    `qmx consolidate` / `session-end` hook calls. These are the code defaults, so the lines are
    optional unless you move or swap the judge (on the fallback, `chat_url =
    "http://spark-0f50.local:8000/v1"`). `chat_timeout = 300` (env `QMX_CHAT_TIMEOUT`) is the
    judge's timeout; sampling, caps and filing rules are the defaults (see *Judge client settings*
    above).
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
