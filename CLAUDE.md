# qmx — Query Memory indeX

**qmx** is a local, private semantic-search engine over **code + chats**. It indexes source
repositories and Claude Code conversation transcripts into a local vector + full-text index and
serves them to agents (and humans) by *meaning*, not just exact text. Everything runs on-device —
proprietary code and conversations never leave the machine.

Derived from [`tobi/qmd`](https://github.com/tobi/qmd) (MIT), reworked to be Qwen-powered and to
add on-the-fly chat capture.

## Goals

- **Semantic search over code** — AST-aware chunking (tree-sitter), find-by-meaning across many repos.
- **Chat memory** — index existing Claude Code transcripts (`~/.claude/projects/*.jsonl`) and capture
  new turns live via a Stop hook. Full-fidelity recall + optional distilled summaries.
- **Local & private** — no code/chat leaves the Mac; nothing embedded to a cloud service.
- **Plugs into Claude Code** — resident MCP server exposing `query` / `get` / `status`.

## Intended stack (finalize after the rewrite-approach decision)

- **Python 3.12**, managed with **`uv`** (never bare `python`/`pip`).
- **Embeddings / rerank / summarize:** the **Qwen** family (Qwen3-Embedding, Qwen3-Reranker, a Qwen
  chat model) — one Apache-2.0 stack. Served locally (Ollama or in-process — TBD by chosen option).
- **Store:** SQLite with `sqlite-vec` (vectors) + FTS5 (BM25). Content-hash tables for incremental
  reindex + dedup.
- **Chunking:** tree-sitter for code; markdown-aware for docs; JSONL→turn parser for chats.
- **Search:** vector + BM25 → Reciprocal Rank Fusion → optional Qwen3-Reranker.
- **Interface:** a resident **MCP server** (not a per-query CLI — pay startup once).

## Conventions

- Python 3.12 + `uv` only. Ruff for lint/format. Type hints throughout.
- The markdown/source files on disk are the **source of truth**; the SQLite index is a
  **rebuildable shadow** — never store data only in the DB.
- Robustness first: incremental reindex (per-file + per-chunk hashing), dedup, tombstone deletes,
  resumable indexing, graceful handling when the model backend is down.

## Repo

- Remote: `https://github.com/the-dsvolk/qmx` (public). Commit as
  `225248328+the-dsvolk@users.noreply.github.com` — the GitHub noreply email for the **the-dsvolk**
  account, so commits attribute to it (not the separate `dsovlk` account) and no real email lands in
  public history. Do **not** use a corporate email.

> Status: scaffolding. The concrete rewrite approach (fork-and-strip vs from-scratch, model hosting)
> is being chosen before code lands.
