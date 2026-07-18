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
from qmx.config import Settings
from qmx.embed import EmbedBackendError, OllamaEmbedder
from qmx.index import backfill_chats, index_memory, index_paths
from qmx.learnings import add_learning, lessons
from qmx.rerank import make_reranker
from qmx.search import search
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
            stats = backfill_chats(projects, store, embedder, force=args.force)
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
    return capture(sys.stdin.read(), settings)


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


def _cmd_lessons(settings: Settings, args: argparse.Namespace) -> int:
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
        print(f"{i:>2}. [{le['score']:.4f}] #{le['learning_id']} ({le['type']}/{scope}) "
              f"imp={le['importance']}")
        print(f"    {le['statement']}")
        if le["detail"]:
            print(f"      ↳ {le['detail']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmx", description="Query Memory indeX")
    parser.add_argument("-v", "--verbose", action="store_true", help="log indexing detail")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show resolved config and index stats")

    p_index = sub.add_parser("index", help="index code file(s) or directory(ies)")
    p_index.add_argument("paths", nargs="+", help="files or directories to index")
    p_index.add_argument("--force", action="store_true", help="re-index unchanged files too")

    p_bf = sub.add_parser("backfill-chats", help="index existing Claude Code transcripts")
    p_bf.add_argument(
        "--projects", default=None, help="transcripts dir (default ~/.claude/projects)"
    )
    p_bf.add_argument("--force", action="store_true", help="re-index unchanged transcripts too")

    sub.add_parser("capture", help="Stop-hook entrypoint: index the transcript named on stdin")

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

    p_les = sub.add_parser("lessons", help="recall distilled lessons (ranked)")
    p_les.add_argument("query", help="what to recall lessons about")
    p_les.add_argument("-k", type=int, default=5, help="number of lessons (default 5)")
    p_les.add_argument(
        "--type", choices=["decision", "mistake", "howto"], default=None, help="filter by type"
    )
    p_les.add_argument("--scope", default=None, help="filter to a repo key (+ global)")
    p_les.add_argument("--json", action="store_true", help="emit JSON instead of text")

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
    "index-memory": _cmd_index_memory,
    "refresh": _cmd_refresh,
    "query": _cmd_query,
    "add-learning": _cmd_add_learning,
    "lessons": _cmd_lessons,
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
