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

Phase 2 (robustness core) landing: tree-sitter code chunking + hybrid **vector + BM25 → RRF**
search, now with **incremental reindex** (only changed chunks re-embed), **cross-file dedup**,
**tombstone deletes**, and a **filesystem watcher** — via `qmx index` / `query` / `watch` / `gc`.
See [`plan/`](./plan) for the full design and phasing.

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

## License

MIT — see [LICENSE](./LICENSE).
