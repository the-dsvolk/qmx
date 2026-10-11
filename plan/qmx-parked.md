# qmx — Parked items ("park this, come back later")

Mid-chat you often hit an idea worth keeping but not worth derailing the current task for. **Parking**
sets it aside in a dedicated, easy-to-list place; when the work is done you close it, and anything
left untouched for too long is offered for review instead of piling up silently.

## How it differs from learnings

| | Learning | Parked item |
|---|---|---|
| Direction | backward — "we found X is true" | forward — "I intend to do X" |
| Lifecycle | lives indefinitely (retire if wrong) | `open` → `done` / `dropped`; meant to go away |
| Surfaced | pushed at SessionStart + pulled by query | **pulled only** — listed when you ask |

## Decisions

- **Name:** *park* (`qmx park`, `qmx parked`, `qmx unpark`; MCP `park` / `list_parked` / `unpark` /
  `resume_parked`).
- **Storage:** `~/.qmx/parked/` (private, per machine; `parked_root` / `QMX_PARKED_ROOT`).
- **Expiry:** soft **stale review** only — never auto-archived or deleted.
- **No SessionStart nudge** — parked items appear only when asked for.

## Storage: files are truth

One markdown file per item, human-readable and hand-editable:

```
~/.qmx/parked/0007-offline-bm25-fallback.md          # open
~/.qmx/parked/archive/0003-rename-cli-verbs.md       # done / dropped
```

```markdown
---
id: 7
status: open
scope: the-dsvolk/qmx
created: 2026-10-10T14:03:00Z
updated: 2026-10-10T14:03:00Z
session: /Users/me/.claude/projects/.../abc.jsonl
---
# Offline BM25 fallback for `lessons`

Why / context: lessons() still needs the embedder; mirror the query degrade path.

## Resolution

(written on `unpark`)
```

- **ids** are monotonic across `parked/` + `archive/` (file created with `O_EXCL` so concurrent
  parks cannot collide).
- **scope** is the canonical repo key (`qmx.scope.canonical_repo_key`), or empty for global. The
  CLI defaults to the current repo; MCP callers pass `cwd` (or an explicit `scope`).
- Closing rewrites the frontmatter (`status`, `closed`, resolution note) and **moves** the file to
  `archive/`. Nothing is ever deleted; `qmx reopen` moves it back.

## Index: a rebuildable shadow

- Each **open** item is embedded as one `kind='parked'` document (`path='parked:<id>'`,
  `repo=<scope>|_global`) so `list_parked(query=…)` ranks by meaning and `park` can report
  *similar* open items (dedup hint, never a block).
- Closing removes the document; reopening re-adds it.
- **Backend down:** `park`/`unpark`/listing are pure file operations and always succeed; the embed
  step is best-effort (`indexed: false`). `qmx refresh` (and `qmx parked --sync`) reconciles the
  index with the files — adds/updates open items, drops anything closed or deleted.

## Stale review

An open item whose `updated` is older than `parked_stale_days` (default 30) is flagged
`stale: true`. `qmx parked --stale` lists them with the two exits: `qmx keep <id>` (bump `updated`,
stay open) or `qmx unpark <id> --drop`.

## Resume

`resume_parked(id)` returns the item plus the best-matching chat turns (`recall` over the item's
text), so picking it back up restores the surrounding conversation, not just the one-liner.

## Surface

| CLI | MCP | Purpose |
|---|---|---|
| `qmx park "text" [--why …] [--global\|--scope K]` | `park(text, why, cwd, scope, session)` | set aside |
| `qmx parked [--all] [--archived] [--stale] [--json] [query]` | `list_parked(cwd, scope, all_scopes, status, query, k)` | list / search |
| `qmx unpark ID [--drop] [--note …]` | `unpark(id, status, note)` | close (archive) |
| `qmx reopen ID` | — | undo a close |
| `qmx keep ID` | — | stale review: still relevant |
| `qmx parked --sync` (also in `qmx refresh`) | — | rebuild shadow index |
| — | `resume_parked(id)` | item + originating chat context |

## Later (not in MVP)

- Auto-suggest close: Stop hook / consolidate matches recent turns + commits against open items and
  *proposes* closing (`#7 looks resolved by deead9d`). Never closes on its own.
- Consolidate pass detects un-parked "let's come back to this" and proposes items.
- A `done` resolution note feeds the learnings tier (where a loop closing becomes a lesson).
