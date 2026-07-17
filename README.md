# qmx — Query Memory indeX

Local, private semantic search over your **code and chats**.

qmx indexes source repositories (AST-aware) and your Claude Code conversation history into an
on-device vector + full-text index, and serves them by *meaning* to agents via MCP — or to you on
the command line. Nothing leaves your machine.

Powered by the [Qwen](https://github.com/QwenLM) embedding/rerank models and
[`sqlite-vec`](https://github.com/asg017/sqlite-vec). Derived from
[`tobi/qmd`](https://github.com/tobi/qmd) (MIT).

## Why

- **Find by meaning, not grep** — "where's the launcher logic" instead of guessing symbol names.
- **Remember conversations** — semantic recall across every past Claude Code session.
- **Private by construction** — proprietary code and chats are embedded and stored locally only.

## Status

Phase 3 (MCP server) landing: qmx now runs as a resident **MCP server** exposing
`query` / `search_code` / `get` / `status` to Claude Code, on top of tree-sitter chunking,
incremental indexing, and hybrid **vector + BM25 → RRF** search (`index` / `query` / `watch` / `gc`).
Reranking is deferred (RRF-only) — Ollama has no rerank endpoint; see
[`plan/qmx-ml-notes.md`](./plan/qmx-ml-notes.md). See [`plan/`](./plan) for the full design.

## Development

Python 3.12 + [`uv`](https://docs.astral.sh/uv/). The model backend (Ollama) runs on the DGX Spark
in prod; point at it with `QMX_OLLAMA_URL` (see [`plan/qmx-deployment.md`](./plan/qmx-deployment.md)).

```bash
uv sync                         # create the venv, install qmx + dev tools
uv run pytest                   # unit tests (live-Ollama tests skip when unreachable)
uv run ruff check . && uv run ruff format --check .
uv run qmx status               # resolved config + index stats

# index code and search it (needs a running Ollama backend):
export QMX_OLLAMA_URL=http://spark-0e81.local:11434
uv run qmx index ~/some/repo
uv run qmx query "where is the retry logic" -k 5
uv run qmx watch ~/some/repo    # keep the index live as files change
uv run qmx gc                   # purge tombstoned chunks

# run the live embed round-trip against the Spark:
uv run pytest -m integration
```

## Use from Claude Code (MCP)

Run the resident server (on the Spark in prod; it owns the index + background loops):

```bash
uv run qmx serve                       # HTTP on 127.0.0.1:8765/mcp by default
QMX_MCP_HOST=0.0.0.0 uv run qmx serve  # bind to the LAN so other machines can reach it
```

Register it with Claude Code — either the CLI:

```bash
claude mcp add --transport http qmx http://spark-0e81.local:8765/mcp
```

…or in `settings.json`:

```json
{ "mcpServers": { "qmx": { "type": "http", "url": "http://spark-0e81.local:8765/mcp" } } }
```

The tools then appear as `mcp__qmx__query`, `mcp__qmx__search_code`, `mcp__qmx__get`,
`mcp__qmx__status`. (`qmx serve --transport stdio` is available for a local, single-client setup.)

## License

MIT — see [LICENSE](./LICENSE).
