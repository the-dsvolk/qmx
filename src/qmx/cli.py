"""qmx CLI — thin admin surface over the store/embed/index/search layers.

Ships ``status`` / ``index`` / ``query`` plus ``watch`` and ``gc`` (Phase 2). Later phases add
``serve``, ``backfill-chats``, and ``capture`` per ``plan/qmx-plan.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from qmx.config import Settings
from qmx.embed import EmbedBackendError, OllamaEmbedder
from qmx.index import index_paths
from qmx.search import search
from qmx.store import Store, StoreSchemaMismatch
from qmx.watch import watch


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


def _cmd_watch(settings: Settings, args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths]
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
    try:
        with _open_store(settings) as store, OllamaEmbedder(settings) as embedder:
            results = search(store, embedder, args.text, k=args.k, kind=args.kind)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmx", description="Query Memory indeX")
    parser.add_argument("-v", "--verbose", action="store_true", help="log indexing detail")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show resolved config and index stats")

    p_index = sub.add_parser("index", help="index code file(s) or directory(ies)")
    p_index.add_argument("paths", nargs="+", help="files or directories to index")
    p_index.add_argument("--force", action="store_true", help="re-index unchanged files too")

    p_query = sub.add_parser("query", help="hybrid (vector + BM25) search")
    p_query.add_argument("text", help="the query text")
    p_query.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p_query.add_argument("--kind", default=None, help="filter by kind (code|doc|chat|learning)")

    p_watch = sub.add_parser("watch", help="watch path(s) and keep the index live")
    p_watch.add_argument("paths", nargs="+", help="files or directories to watch")

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
    "query": _cmd_query,
    "watch": _cmd_watch,
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
