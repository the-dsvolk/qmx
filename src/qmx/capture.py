"""``qmx capture`` — the Claude Code Stop-hook entrypoint (the "write door").

The hook feeds JSON on stdin (`transcript_path`, `session_id`, `cwd`, `hook_event_name`); we
incrementally index that transcript. It is **best-effort and must never fail a turn**: any error is
swallowed and we exit 0. Cheap because re-indexing a transcript only embeds the newest turn(s)
(per-chunk dedup handles the rest — see :func:`qmx.index.index_transcript`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qmx.chunk.chat import ChatSource
from qmx.config import Settings
from qmx.embed import OllamaEmbedder
from qmx.index import index_memory_dir, index_transcript
from qmx.store import Store

log = logging.getLogger("qmx.capture")


def capture(stdin_text: str, settings: Settings, source: ChatSource = "claude") -> int:
    """Index the transcript named in the hook payload. Always returns 0 (never blocks a turn).

    ``source`` selects the transcript schema and is set explicitly by the caller (the Claude Code
    Stop hook uses the default ``"claude"``; the Cursor ``stop`` hook passes ``"cursor"``).
    """
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
        transcript_path = payload.get("transcript_path")
        if not transcript_path:
            return 0
        with (
            Store.open(settings.db_path, settings.embed_dim, settings.embed_model) as store,
            OllamaEmbedder(settings) as embedder,
        ):
            stats = index_transcript(transcript_path, store, embedder, source=source)
            # Also refresh this project's curated memory (sibling `memory/` of the transcript).
            mem = index_memory_dir(Path(transcript_path).parent / "memory", store, embedder)
        log.info(
            "captured %s (chat +%d, memory +%d)",
            transcript_path,
            stats.chunks_embedded,
            mem.chunks_embedded,
        )
    except Exception as exc:  # noqa: BLE001 — a hook must not break the turn on any failure
        log.warning("capture skipped: %s", exc)
    return 0
