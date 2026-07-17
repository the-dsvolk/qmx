"""qmx CLI — thin admin surface over the store/embed layer.

Phase 0 ships ``qmx status`` (index stats + resolved config). Later phases add ``index``,
``query``, ``serve``, ``backfill-chats``, ``capture`` per ``plan/qmx-plan.md``.
"""

from __future__ import annotations

import argparse
import json
import sys

from qmx.config import Settings
from qmx.store import Store, StoreSchemaMismatch


def _cmd_status(settings: Settings) -> int:
    info: dict[str, object] = {"config": settings.as_dict()}
    try:
        with Store.open(settings.db_path, settings.embed_dim, settings.embed_model) as store:
            info["index"] = store.counts()
    except StoreSchemaMismatch as exc:
        info["index_error"] = str(exc)
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmx", description="Query Memory indeX")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show resolved config and index stats")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    if args.command == "status":
        return _cmd_status(settings)
    return 1


if __name__ == "__main__":
    sys.exit(main())
