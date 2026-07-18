"""Session hook entrypoints — proactive lesson injection + end-of-session consolidation.

- ``qmx session-start`` (Claude Code ``SessionStart``, matcher ``startup``): resolve the repo scope
  from ``cwd``, pick the top scope-matched + global lessons (importance×recency, budgeted), and emit
  them as ``hookSpecificOutput.additionalContext`` — the documented context-injection field (max
  10k chars). Not stdout, which is *not* injected.
- ``qmx session-end`` (Claude Code ``SessionEnd``): spawn ``qmx consolidate`` **detached** and
  return immediately. SessionEnd blocks session closure until the hook returns (600s), so the heavy
  Qwen pass must not run inline.

Both are best-effort: any error is swallowed and we exit 0 (a hook must never break the session).
See ``plan/qmx-learnings.md`` (*Triggers & wiring*) and the verified hook contract.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from qmx.config import Settings
from qmx.learnings import inject_lessons, render_lesson
from qmx.scope import canonical_repo_key
from qmx.store import Store

log = logging.getLogger("qmx.session")

INJECT_CHAR_BUDGET = 10_000  # SessionStart additionalContext hard cap
_HEADER = "Relevant lessons from past sessions (qmx). Apply them; supersede if now wrong:"


def build_injection(
    store: Store, scope: str | None, *, char_budget: int = INJECT_CHAR_BUDGET
) -> str:
    """The ``additionalContext`` block for a scope: header + rendered lessons, or ``""`` if none."""
    body_budget = char_budget - len(_HEADER) - 1
    lessons = inject_lessons(store, scope, char_budget=max(0, body_budget))
    if not lessons:
        return ""
    lines = "\n".join(render_lesson(le) for le in lessons)
    return f"{_HEADER}\n{lines}"


def session_start(stdin_text: str, settings: Settings) -> str:
    """Return the JSON a ``SessionStart`` hook should print (``""`` = inject nothing)."""
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
        # Only inject on a fresh start, not resume/compact (avoid re-injecting mid-session).
        if payload.get("source") not in (None, "startup", "clear"):
            return ""
        scope = canonical_repo_key(payload.get("cwd") or Path.cwd())
        with Store.open(settings.db_path, settings.embed_dim, settings.embed_model) as store:
            context = build_injection(store, scope)
        if not context:
            return ""
        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    except Exception as exc:  # noqa: BLE001 — a hook must never break the session
        log.warning("session-start inject skipped: %s", exc)
        return ""


def session_end(stdin_text: str, settings: Settings) -> bool:
    """Spawn a detached ``qmx consolidate`` for the ending session. Returns whether it launched."""
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
        transcript = payload.get("transcript_path")
        if not transcript:
            return False
        scope = canonical_repo_key(payload.get("cwd") or Path.cwd())
        cmd = [sys.executable, "-m", "qmx.cli", "consolidate", "--session", transcript]
        if scope:
            cmd += ["--scope", scope]
        subprocess.Popen(  # detached: never blocks session close (SessionEnd is synchronous)
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("session-end consolidate skipped: %s", exc)
        return False
