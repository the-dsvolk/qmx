# qmx — Architecture & the Three Capabilities

qmx delivers three capabilities on **one shared pipeline** (chunk → Qwen-embed via Ollama →
sqlite store → search). They differ mostly in *what* is indexed and *which* query path runs.

1. **Quick semantic search over code** — find code by meaning, fast.
2. **Memory recall** — "when did we discuss X, and what did we conclude?"
3. **Chat summarization / learning** — distill past chats into lessons: mistakes made, better ways
   to do things. (Borrows the *always-on-memory-agent* Ingest→Consolidate→Query pattern.)

## Unified data flow

```mermaid
flowchart TB
  subgraph SRC["Sources"]
    C["Code repos<br/>(xtorch + 7 others)"]
    B["Chat backfill<br/>~/.claude/projects/*.jsonl"]
    H["Live capture<br/>Claude Code <b>Stop hook</b>"]
  end

  subgraph ING["Ingest + chunk"]
    CC["tree-sitter<br/>AST code chunks"]
    CH["jsonl → clean turns<br/>(drop tool payloads)"]
  end

  subgraph OLL["Ollama — Qwen only"]
    E["qwen3-embedding"]
    R["qwen3-reranker"]
    SUM["qwen3 (chat)<br/>summarize / extract"]
  end

  subgraph STORE["Local SQLite store"]
    V[("sqlite-vec<br/>vectors")]
    F[("FTS5<br/>BM25")]
    L[("distilled memory<br/>decisions · mistakes · how-to")]
  end

  CON["Consolidation loop<br/>(periodic Qwen pass)"]
  MCP["qmx MCP server<br/>query · get · status"]
  USER["Claude Code / you"]

  C --> CC
  B --> CH
  H --> CH
  CC --> E
  CH --> E
  CC --> F
  CH --> F
  E --> V

  CH --> CON
  CON --> SUM
  SUM --> L

  V --> MCP
  F --> MCP
  L --> MCP
  R -. "rerank top-k" .-> MCP
  MCP --> USER
```

## How each capability is served

```mermaid
flowchart LR
  Q["query text"]

  subgraph CAP1["1 · Code search"]
    Q1["vector(code) + BM25<br/>→ RRF → rerank"]
    R1["ranked code chunks<br/>file:line"]
  end
  subgraph CAP2["2 · Memory recall"]
    Q2["vector(chat) + BM25<br/>filter kind=chat"]
    R2["matching turns +<br/>expand to conclusion"]
  end
  subgraph CAP3["3 · Learnings"]
    Q3["vector(distilled)<br/>filter kind=learning"]
    R3["lessons / mistakes /<br/>better-way notes"]
  end

  Q --> Q1 --> R1
  Q --> Q2 --> R2
  Q --> Q3 --> R3
```

| Capability | Indexed content | Ingest path | Query path | Answers |
|---|---|---|---|---|
| **1 · Code search** | code chunks (`kind=code`) | tree-sitter → embed → vec+FTS | vector+BM25 → RRF → rerank | "where's the launcher logic" → `xtorch.py:591` |
| **2 · Memory recall** | chat turns (`kind=chat`) | jsonl/live → clean → embed → vec+FTS | vector+BM25, `kind=chat`, expand hit → surrounding turns | "when did we discuss local embeddings, and the conclusion?" → the turns + the decision |
| **3 · Learnings** | distilled notes (`kind=learning`) | **consolidation loop**: Qwen reads recent chats → extracts decisions/mistakes/how-to → dedups/updates → stores | vector over `kind=learning` | "how should I raise a gcp-infra IAM PR" → "project-level, ask in #platform-security-support (learned 2026-07)" |

## Capability #3 — the "learning" layer (always-on-memory pattern, localized)

The always-on-memory-agent does **Ingest → Consolidate → Query**; #3 adapts it to local Qwen:

- **Ingest** (cheap, per-turn): the Stop hook already writes clean turn markdown. Tag salient turns.
- **Consolidate** (periodic — a timer or on-demand, mimicking "sleep consolidation"): a Qwen pass
  reads recent/unconsolidated chats and extracts structured **learnings**:
  - `decision` — what we concluded and why
  - `mistake` — what went wrong + the correction (→ "learn from past mistakes")
  - `howto` — a better/repeatable way to do a task (→ "do things better")
  - each with `topic`, `importance`, `source session anchors`.
  - **Dedup/update:** if a learning matches an existing one, update it (don't duplicate) — the
    consolidation step merges and compresses, exactly like the reference agent's insight-generation.
- **Query** (retrieval-time): learnings are embedded and searchable; the MCP `query` can return them
  alongside code/chat hits, or an agent can pull them at session start.

This is the tier that turns raw recall (#2) into *usable* lessons (#3). It complements your existing
curated `~/.claude/.../memory/` files — qmx auto-generates candidate learnings; the curated files
stay the hand-picked canon.

> The parallel research pass on the always-on-memory-agent will refine this section (extraction
> triggers, importance scoring, consolidation cadence, memory-type taxonomy) before Phase 4.

## Where this sits in the plan

- Capabilities **1 & 2** are delivered by Phases 0–4 of [qmx-plan.md](./qmx-plan.md) (store → code
  slice → robustness → MCP → chats).
- Capability **3** (consolidation/learnings) is an addition to Phase 4/5: the `consolidate` step +
  `kind=learning` documents + a periodic trigger.
