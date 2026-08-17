"""qmx CLI — thin admin surface over the store/embed/index/search layers.

Ships ``status`` / ``index`` / ``query`` / ``watch`` / ``sources`` / ``remove`` / ``gc`` /
``serve``, plus ``backfill-chats`` and ``capture`` (chat memory) per ``plan/qmx-plan.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from qmx.capture import capture
from qmx.chat import ChatBackendError, OllamaChat
from qmx.config import Settings
from qmx.consolidate import consolidate_session
from qmx.embed import EmbedBackendError, OllamaEmbedder
from qmx.index import backfill_chats, index_memory, index_paths, index_transcript
from qmx.learnings import (
    add_learning,
    deprecate_learning,
    learning_to_dict,
    lessons,
    normalize_importance,
    restore_learning,
    update_learning,
)
from qmx.promote import PromotionError, promotable, promote
from qmx.rerank import make_reranker
from qmx.search import search
from qmx.session import session_end, session_start
from qmx.store import Store, StoreSchemaMismatch
from qmx.watch import watch

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _open_store(settings: Settings) -> Store:
    return Store.open(settings.db_path, settings.embed_dim, settings.embed_model)


def _cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    info: dict[str, object] = {"config": settings.as_dict()}
    try:
        with _open_store(settings) as store:
            info["index"] = store.index_stats()
    except StoreSchemaMismatch as exc:
        info["index_error"] = str(exc)
    print(json.dumps(info, indent=2))
    return 0


def _cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"no such path(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            stats = index_paths(paths, store, embedder, force=args.force)
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"index failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"indexed {stats.files_indexed} file(s): {stats.chunks_embedded} embedded, "
        f"{stats.chunks_reused} reused; removed {stats.files_removed} deleted, "
        f"orphaned {stats.chunks_orphaned}; skipped {stats.files_skipped}, "
        f"scanned {stats.files_scanned}"
    )
    for err in stats.errors:
        print(f"  ! {err}", file=sys.stderr)
    return 0


def _cmd_backfill_chats(settings: Settings, args: argparse.Namespace) -> int:
    projects = Path(args.projects) if args.projects else DEFAULT_PROJECTS_DIR
    if not projects.exists():
        print(f"no such projects dir: {projects}", file=sys.stderr)
        return 2
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            stats = backfill_chats(projects, store, embedder, force=args.force, source=args.source)
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"backfill-chats failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"indexed {stats.files_indexed} transcript(s): {stats.chunks_embedded} turns embedded, "
        f"{stats.chunks_reused} reused; skipped {stats.files_skipped}, "
        f"scanned {stats.files_scanned}"
    )
    for err in stats.errors:
        print(f"  ! {err}", file=sys.stderr)
    return 0


def _cmd_capture(settings: Settings, args: argparse.Namespace) -> int:
    # Stop-hook entrypoint: hook JSON arrives on stdin. Best-effort; never fails a turn.
    return capture(sys.stdin.read(), settings, source=args.source)


def _cmd_session_start(settings: Settings, args: argparse.Namespace) -> int:
    # SessionStart hook: emit hookSpecificOutput.additionalContext JSON (or nothing). Never fails.
    out = session_start(sys.stdin.read(), settings)
    if out:
        print(out)
    return 0


def _cmd_session_end(settings: Settings, args: argparse.Namespace) -> int:
    # SessionEnd hook: spawn a detached consolidate so it never blocks session close. Never fails.
    session_end(sys.stdin.read(), settings)
    return 0


def _cmd_refresh(settings: Settings, args: argparse.Namespace) -> int:
    """Sync the flat KB from all configured sources: code_roots + chats + memory."""
    roots = [Path(r).expanduser() for r in settings.code_roots]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        print(f"code_roots not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            code = index_paths(roots, store, embedder, force=args.force)
            chats = backfill_chats(DEFAULT_PROJECTS_DIR, store, embedder, force=args.force)
            mem = index_memory(settings.memory_globs, store, embedder, force=args.force)
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"refresh failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"code:   {code.files_indexed} files, {code.chunks_embedded} embedded "
        f"({len(roots)} root(s))\n"
        f"chats:  {chats.files_indexed} transcripts, {chats.chunks_embedded} turns embedded\n"
        f"memory: {mem.files_indexed} files, {mem.chunks_embedded} embedded"
    )
    for err in (*code.errors, *chats.errors, *mem.errors):
        print(f"  ! {err}", file=sys.stderr)
    return 0


def _cmd_index_memory(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            stats = index_memory(settings.memory_globs, store, embedder, force=args.force)
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"index-memory failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"indexed {stats.files_indexed} memory file(s): {stats.chunks_embedded} embedded, "
        f"{stats.chunks_reused} reused; skipped {stats.files_skipped}, "
        f"scanned {stats.files_scanned}"
    )
    for err in stats.errors:
        print(f"  ! {err}", file=sys.stderr)
    return 0


def _watch_targets(settings: Settings, arg_paths: list[str]) -> list[Path]:
    """Paths to watch: the CLI args, or the configured ``code_roots`` when none are given."""
    raw = arg_paths or [str(Path(r).expanduser()) for r in settings.code_roots]
    return [Path(p) for p in raw]


def _cmd_watch(settings: Settings, args: argparse.Namespace) -> int:
    paths = _watch_targets(settings, args.paths)
    if not paths:
        print("nothing to watch: pass path(s) or set code_roots in config", file=sys.stderr)
        return 2
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"no such path(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            print(f"watching {', '.join(str(p) for p in paths)} — Ctrl-C to stop")
            watch(paths, store, embedder)
    except StoreSchemaMismatch as exc:
        print(f"watch failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_gc(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store:
            purged = store.purge_orphans()
    except StoreSchemaMismatch as exc:
        print(f"gc failed: {exc}", file=sys.stderr)
        return 1
    print(f"purged {purged} tombstoned chunk(s)")
    return 0


def _cmd_sources(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store:
            sources = store.list_sources()
    except StoreSchemaMismatch as exc:
        print(f"sources failed: {exc}", file=sys.stderr)
        return 1
    if not sources:
        print("(nothing indexed)")
        return 0
    width = max(len(s["repo"] or "?") for s in sources)
    for s in sources:
        print(
            f"{(s['repo'] or '?'):<{width}}  {s['documents']:>5} files  "
            f"{s['chunks']:>6} chunks  {s['sample_path']}"
        )
    return 0


def _cmd_remove(settings: Settings, args: argparse.Namespace) -> int:
    target = str(Path(args.path).resolve())
    try:
        with _open_store(settings) as store:
            docs, orphaned = store.remove_source(target)
    except StoreSchemaMismatch as exc:
        print(f"remove failed: {exc}", file=sys.stderr)
        return 1
    if docs == 0:
        print(f"nothing indexed under {target}")
        return 0
    print(
        f"removed {docs} document(s), orphaned {orphaned} chunk(s) — run `qmx gc` to reclaim space"
    )
    return 0


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    from qmx.mcp_server import serve  # deferred: pulls in the mcp SDK only when serving

    transport = "stdio" if args.transport == "stdio" else "streamable-http"
    host = args.host or settings.mcp_host
    port = args.port or settings.mcp_port
    if transport == "stdio":
        print("qmx MCP server on stdio", file=sys.stderr)
    else:
        print(f"qmx MCP server on http://{host}:{port}/mcp", file=sys.stderr)
    serve(settings, transport=transport, host=host, port=port)
    return 0


def _cmd_query(settings: Settings, args: argparse.Namespace) -> int:
    reranker = make_reranker(settings)
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            results = search(
                store, embedder, args.text, k=args.k, kind=args.kind, reranker=reranker
            )
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("(no results)")
        return 0
    for i, r in enumerate(results, 1):
        h = r.hit
        loc = h.path or f"doc#{h.doc_id}"
        if h.start_line is not None:
            loc = f"{loc}:{h.start_line}"
        sym = f" {h.symbol}" if h.symbol else ""
        head = h.text.strip().splitlines()[0][:100] if h.text.strip() else ""
        print(f"{i:>2}. [{r.score:.4f}] {loc}{sym}")
        print(f"    {head}")
    return 0


def _cmd_add_learning(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            learning_id = add_learning(
                store,
                embedder,
                type=args.type,
                statement=args.statement,
                topic=args.topic,
                scope=args.scope,
                detail=args.detail,
                importance=args.importance,
            )
    except (StoreSchemaMismatch, EmbedBackendError, ValueError) as exc:
        print(f"add-learning failed: {exc}", file=sys.stderr)
        return 1
    print(f"added learning #{learning_id} [{args.type}]: {args.statement}")
    return 0


def _cmd_update_learning(settings: Settings, args: argparse.Namespace) -> int:
    """Fix a lesson in place. ``--clear-*``/``--global`` NULL a field; omitted fields are kept."""
    patch: dict = {}
    for field in ("type", "topic", "scope", "statement", "detail", "importance"):
        value = getattr(args, field)
        if value is not None:
            patch[field] = value
    if args.clear_detail:
        patch["detail"] = None
    if args.clear_topic:
        patch["topic"] = None
    if args.make_global:
        patch["scope"] = None
    if not patch:
        print("update-learning: nothing to change (pass at least one field)", file=sys.stderr)
        return 2
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            learning = update_learning(store, embedder, args.id, **patch)
    except (StoreSchemaMismatch, EmbedBackendError, ValueError) as exc:
        print(f"update-learning failed: {exc}", file=sys.stderr)
        return 1
    if learning is None:
        print(f"update-learning: no learning #{args.id}", file=sys.stderr)
        return 1
    print(f"updated learning #{learning.learning_id}: {', '.join(sorted(patch))}")
    print(f"  [{learning.type}/{learning.scope or 'global'}] imp={learning.importance:.2f} "
          f"{learning.statement}")
    return 0


def _cmd_deprecate_learning(settings: Settings, args: argparse.Namespace) -> int:
    """Soft-retire a lesson (optionally naming its replacement); keeps the row for audit."""
    try:
        with _open_store(settings) as store:
            learning = deprecate_learning(
                store, args.id, reason=args.reason, superseded_by=args.superseded_by
            )
    except (StoreSchemaMismatch, ValueError) as exc:
        print(f"deprecate-learning failed: {exc}", file=sys.stderr)
        return 1
    if learning is None:
        print(f"deprecate-learning: no learning #{args.id}", file=sys.stderr)
        return 1
    tail = f" (superseded by #{learning.superseded_by})" if learning.superseded_by else ""
    print(f"deprecated learning #{learning.learning_id}{tail}: {learning.statement}")
    if learning.promoted_to:
        print(f"  note: already promoted to {learning.promoted_to} — edit that file too")
    return 0


def _cmd_restore_learning(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store:
            learning = restore_learning(store, args.id)
    except StoreSchemaMismatch as exc:
        print(f"restore-learning failed: {exc}", file=sys.stderr)
        return 1
    if learning is None:
        print(f"restore-learning: no learning #{args.id}", file=sys.stderr)
        return 1
    print(f"restored learning #{learning.learning_id}: {learning.statement}")
    return 0


def _cmd_fix_importance(settings: Settings, args: argparse.Namespace) -> int:
    """Rescale `importance` values above 1.0 (model 1-5/1-10 answers) onto the documented 0..1.

    Reports by default and only writes with ``--apply``: learnings live *only* in the DB (unlike
    code/chats, which are a rebuildable shadow of files on disk), so a bulk rewrite is not something
    to do implicitly. ``updated_at`` is left untouched so a repair does not reorder recall.
    """
    try:
        with _open_store(settings) as store:
            broken = [le for le in store.list_learnings(live_only=False) if le.importance > 1.0]
            total = len(store.list_learnings(live_only=False))
            if not broken:
                print(f"all {total} learning(s) already within 0..1 — nothing to fix")
                return 0
            print(f"{len(broken)} of {total} learning(s) have importance > 1.0:")
            for le in sorted(broken, key=lambda le: le.importance, reverse=True)[: args.show]:
                print(f"  #{le.learning_id} {le.importance:g} -> "
                      f"{normalize_importance(le.importance):.2f}  {le.statement[:60]}")
            if len(broken) > args.show:
                print(f"  … and {len(broken) - args.show} more")
            if not args.apply:
                print("dry run — re-run with --apply to write (back up ~/.qmx/index.db first)")
                return 0
            fixed = sum(
                store.set_learning_importance(le.learning_id, normalize_importance(le.importance))
                for le in broken
            )
    except StoreSchemaMismatch as exc:
        print(f"fix-importance failed: {exc}", file=sys.stderr)
        return 1
    print(f"rescaled {fixed} learning(s) onto 0..1 (updated_at untouched)")
    return 0


def _cmd_lessons(settings: Settings, args: argparse.Namespace) -> int:
    if args.review:
        return _cmd_lessons_review(settings, args)
    if args.deprecated:
        return _cmd_lessons_deprecated(settings, args)
    if not args.query:
        print("lessons: pass a query, or --review / --deprecated", file=sys.stderr)
        return 2
    reranker = make_reranker(settings)
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            results = lessons(
                store,
                embedder,
                args.query,
                k=args.k,
                type=args.type,
                scope=args.scope,
                include_retired=args.include_retired,
                reranker=reranker,
            )
    except (StoreSchemaMismatch, EmbedBackendError) as exc:
        print(f"lessons failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    if not results:
        print("(no lessons)")
        return 0
    for i, le in enumerate(results, 1):
        scope = le["scope"] or "global"
        retired = " RETIRED" if le.get("deprecated_at") else ""
        print(f"{i:>2}. [{le['score']:.4f}] #{le['learning_id']} ({le['type']}/{scope}) "
              f"imp={le['importance']}{retired}")
        print(f"    {le['statement']}")
        if le["detail"]:
            print(f"      ↳ {le['detail']}")
        if retired:
            print(f"      ✗ {_retired_note(le)}")
    return 0


def _cmd_lessons_deprecated(settings: Settings, args: argparse.Namespace) -> int:
    """List soft-retired lessons with their reason + replacement breadcrumb."""
    try:
        with _open_store(settings) as store:
            retired = store.list_learnings(
                scope=args.scope, include_global=True, deprecated_only=True, limit=args.k
            )
    except StoreSchemaMismatch as exc:
        print(f"lessons --deprecated failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([learning_to_dict(le) for le in retired], indent=2))
        return 0
    if not retired:
        print("(no deprecated lessons)")
        return 0
    print(f"{len(retired)} deprecated lesson(s) — `qmx restore-learning <id>` to revive:")
    for le in retired:
        print(f"  #{le.learning_id} [{le.type}/{le.scope or 'global'}] {le.statement}")
        print(f"      ✗ {_retired_note(learning_to_dict(le))}")
    return 0


def _retired_note(le: dict) -> str:
    """``deprecated <when>: <reason> -> superseded by #N`` for a retired lesson dict."""
    parts = [f"deprecated {le.get('deprecated_at') or '?'}"]
    if le.get("deprecated_reason"):
        parts.append(str(le["deprecated_reason"]))
    note = ": ".join(parts)
    if le.get("superseded_by"):
        note += f" → superseded by #{le['superseded_by']}"
    return note


def _cmd_lessons_review(settings: Settings, args: argparse.Namespace) -> int:
    """List promotion-eligible lessons (live, unpromoted, over the gate) for `qmx promote`."""
    try:
        with _open_store(settings) as store:
            eligible = promotable(
                store, min_importance=args.min_importance, min_reuse=args.min_reuse
            )
    except StoreSchemaMismatch as exc:
        print(f"lessons --review failed: {exc}", file=sys.stderr)
        return 1
    if not eligible:
        print("(no lessons eligible for promotion)")
        return 0
    print(f"{len(eligible)} lesson(s) eligible for promotion — `qmx promote <id>`:")
    for le in eligible:
        scope = le.scope or "global"
        print(f"  #{le.learning_id} [{le.type}/{scope}] imp={le.importance:.2f} "
              f"reuse={le.reuse_count}: {le.statement}")
    return 0


def _cmd_promote(settings: Settings, args: argparse.Namespace) -> int:
    try:
        with _open_store(settings) as store:
            path = promote(store, args.id, memory_root=settings.promoted_memory_root)
    except (StoreSchemaMismatch, PromotionError) as exc:
        print(f"promote failed: {exc}", file=sys.stderr)
        return 1
    print(f"promoted learning #{args.id} -> {path}")
    return 0


def _cmd_consolidate(settings: Settings, args: argparse.Namespace) -> int:
    """Distil chat turns into learnings — one session (--session) or every chat doc (--all)."""
    if not args.session and not args.all:
        print("consolidate: pass --session <transcript> or --all", file=sys.stderr)
        return 2
    try:
        with (
            _open_store(settings) as store,
            OllamaEmbedder(settings) as embedder,
            OllamaChat(settings) as chat,
        ):
            targets: list[int] = []
            if args.session:
                path_key = str(Path(args.session).resolve())
                if store.document_id("chat", path_key) is None:
                    index_transcript(Path(args.session), store, embedder)  # index if new
                doc_id = store.document_id("chat", path_key)
                if doc_id is None:
                    print(f"no chat turns indexed for {args.session}", file=sys.stderr)
                    return 1
                targets = [doc_id]
            else:
                targets = [doc_id for doc_id, _ in store.list_documents("chat")]

            created = updated = superseded = candidates = 0
            deprecated = dropped = 0
            for doc_id in targets:
                res = consolidate_session(store, embedder, chat, doc_id, scope=args.scope)
                created += res.created
                updated += res.updated
                superseded += res.superseded
                deprecated += res.deprecated
                dropped += res.dropped
                candidates += res.candidates
    except (StoreSchemaMismatch, EmbedBackendError, ChatBackendError) as exc:
        print(f"consolidate failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"consolidated {len(targets)} session(s): {candidates} candidate(s) -> "
        f"{created} new, {updated} updated, {superseded} superseded, "
        f"{deprecated} deprecated, {dropped} dropped"
    )
    if dropped:
        # Never let a discarded candidate look like one that was simply never extracted.
        print(f"  ({dropped} candidate(s) dropped as re-learning retired lessons; -v for which)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmx", description="Query Memory indeX")
    parser.add_argument("-v", "--verbose", action="store_true", help="log indexing detail")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show resolved config and index stats")

    p_index = sub.add_parser("index", help="index code file(s) or directory(ies)")
    p_index.add_argument("paths", nargs="+", help="files or directories to index")
    p_index.add_argument("--force", action="store_true", help="re-index unchanged files too")

    p_bf = sub.add_parser("backfill-chats", help="index existing Claude Code / Cursor transcripts")
    p_bf.add_argument(
        "--projects", default=None, help="transcripts dir (default ~/.claude/projects)"
    )
    p_bf.add_argument("--force", action="store_true", help="re-index unchanged transcripts too")
    p_bf.add_argument(
        "--source",
        choices=("claude", "cursor"),
        default="claude",
        help="transcript schema to parse (default: claude)",
    )

    p_cap = sub.add_parser(
        "capture", help="Stop-hook entrypoint: index the transcript named on stdin"
    )
    p_cap.add_argument(
        "--source",
        choices=("claude", "cursor"),
        default="claude",
        help="transcript schema to parse (default: claude; Cursor stop hook passes cursor)",
    )

    sub.add_parser("session-start", help="SessionStart hook: inject relevant lessons (stdin JSON)")
    sub.add_parser("session-end", help="SessionEnd hook: detached consolidate (stdin JSON)")

    p_mem = sub.add_parser("index-memory", help="index Claude memory files (kind=memory)")
    p_mem.add_argument("--force", action="store_true", help="re-index unchanged memory files too")

    p_refresh = sub.add_parser(
        "refresh", help="sync the flat KB: configured code_roots + chats + memory"
    )
    p_refresh.add_argument("--force", action="store_true", help="re-index unchanged files too")

    p_query = sub.add_parser("query", help="hybrid (vector + BM25) search")
    p_query.add_argument("text", help="the query text")
    p_query.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p_query.add_argument("--kind", default=None, help="filter by kind (code|doc|chat|learning)")

    p_add = sub.add_parser("add-learning", help="record a distilled lesson (kind=learning)")
    p_add.add_argument("statement", help="the lesson, one crisp sentence")
    p_add.add_argument(
        "--type", choices=["decision", "mistake", "howto"], required=True, help="lesson type"
    )
    p_add.add_argument("--detail", default=None, help="why / the correction / the better way")
    p_add.add_argument("--topic", default=None, help="short slug for filtering/injection")
    p_add.add_argument("--scope", default=None, help="repo key it applies to (omit = global)")
    p_add.add_argument("--importance", type=float, default=0.5, help="0..1 (default 0.5)")

    p_upd = sub.add_parser("update-learning", help="fix an existing lesson in place")
    p_upd.add_argument("id", type=int, help="learning id")
    p_upd.add_argument("--statement", default=None, help="replacement statement")
    p_upd.add_argument("--detail", default=None, help="replacement detail")
    p_upd.add_argument("--topic", default=None, help="replacement topic slug")
    p_upd.add_argument(
        "--type", choices=["decision", "mistake", "howto"], default=None, help="fix the type"
    )
    p_upd.add_argument("--scope", default=None, help="re-scope to this repo key")
    p_upd.add_argument(
        "--importance", type=float, default=None, help="0..1 — re-weight (no re-embed needed)"
    )
    p_upd.add_argument("--clear-detail", action="store_true", help="drop the detail")
    p_upd.add_argument("--clear-topic", action="store_true", help="drop the topic")
    p_upd.add_argument(
        "--global", dest="make_global", action="store_true", help="clear scope (make it global)"
    )

    p_dep = sub.add_parser("deprecate-learning", help="soft-retire a lesson (reversible)")
    p_dep.add_argument("id", type=int, help="learning id to retire")
    p_dep.add_argument("--reason", default=None, help="why it is being retired")
    p_dep.add_argument(
        "--superseded-by", type=int, default=None, help="id of the lesson replacing it (optional)"
    )

    p_res = sub.add_parser("restore-learning", help="un-retire a lesson deprecated by mistake")
    p_res.add_argument("id", type=int, help="learning id to revive")

    p_fix = sub.add_parser(
        "fix-importance", help="rescale out-of-range importance (>1) onto 0..1; dry run by default"
    )
    p_fix.add_argument("--apply", action="store_true", help="write the changes (default: report)")
    p_fix.add_argument("--show", type=int, default=10, help="how many examples to list (def 10)")

    p_con = sub.add_parser("consolidate", help="distil chat turns into learnings (Qwen)")
    p_con.add_argument("--session", default=None, help="a transcript .jsonl to consolidate")
    p_con.add_argument("--all", action="store_true", help="consolidate every indexed chat doc")
    p_con.add_argument("--scope", default=None, help="repo key to tag the learnings with")

    p_les = sub.add_parser("lessons", help="recall distilled lessons (ranked) or --review")
    p_les.add_argument("query", nargs="?", default=None, help="what to recall lessons about")
    p_les.add_argument("-k", type=int, default=5, help="number of lessons (default 5)")
    p_les.add_argument(
        "--type", choices=["decision", "mistake", "howto"], default=None, help="filter by type"
    )
    p_les.add_argument("--scope", default=None, help="filter to a repo key (+ global)")
    p_les.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p_les.add_argument(
        "--review", action="store_true", help="list promotion-eligible lessons instead"
    )
    p_les.add_argument(
        "--include-retired", action="store_true", help="also match deprecated/superseded lessons"
    )
    p_les.add_argument(
        "--deprecated", action="store_true", help="list soft-retired lessons + why, instead"
    )
    p_les.add_argument("--min-importance", type=float, default=0.6, help="review gate (def 0.6)")
    p_les.add_argument("--min-reuse", type=int, default=1, help="review gate (default 1)")

    p_prom = sub.add_parser("promote", help="graduate a lesson to per-repo curated memory")
    p_prom.add_argument("id", type=int, help="learning id (from `qmx lessons --review`)")

    p_watch = sub.add_parser("watch", help="watch path(s) (or code_roots) and keep the index live")
    p_watch.add_argument(
        "paths", nargs="*", help="files/directories to watch (default: config code_roots)"
    )

    sub.add_parser("sources", help="list indexed sources (grouped by repo)")

    p_remove = sub.add_parser("remove", help="remove a file or directory subtree from the index")
    p_remove.add_argument("path", help="file or directory to drop from the index")

    sub.add_parser("gc", help="purge tombstoned (unreferenced) chunks")

    p_serve = sub.add_parser("serve", help="run the resident MCP server")
    p_serve.add_argument(
        "--transport", choices=["http", "stdio"], default="http", help="default: http"
    )
    p_serve.add_argument("--host", default=None, help="bind host (default from config)")
    p_serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")

    return parser


_COMMANDS = {
    "status": _cmd_status,
    "index": _cmd_index,
    "backfill-chats": _cmd_backfill_chats,
    "capture": _cmd_capture,
    "session-start": _cmd_session_start,
    "session-end": _cmd_session_end,
    "index-memory": _cmd_index_memory,
    "refresh": _cmd_refresh,
    "query": _cmd_query,
    "add-learning": _cmd_add_learning,
    "update-learning": _cmd_update_learning,
    "deprecate-learning": _cmd_deprecate_learning,
    "restore-learning": _cmd_restore_learning,
    "fix-importance": _cmd_fix_importance,
    "lessons": _cmd_lessons,
    "promote": _cmd_promote,
    "consolidate": _cmd_consolidate,
    "watch": _cmd_watch,
    "sources": _cmd_sources,
    "remove": _cmd_remove,
    "gc": _cmd_gc,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    settings = Settings.load()
    return _COMMANDS[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
