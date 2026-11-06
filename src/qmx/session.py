"""Session hook entrypoints — proactive lesson injection + end-of-session consolidation.

- ``qmx session-start`` (Claude Code ``SessionStart``, matcher ``startup``): resolve the repo scope
  from ``cwd``, pick the repo's own lessons first and then the global lessons most similar to the
  repo's ``CLAUDE.md``/``README.md`` head (budgeted), and emit them as
  ``hookSpecificOutput.additionalContext`` — the documented context-injection field (max 10k
  chars). Not stdout, which is *not* injected.
- ``qmx session-end`` (Claude Code ``SessionEnd``): spawn ``qmx consolidate`` **detached** and
  return immediately. SessionEnd blocks session closure until the hook returns (600s), so the heavy
  Qwen pass must not run inline.

Both are best-effort: any error is swallowed and we exit 0 (a hook must never break the session).
See ``plan/qmx-learnings.md`` (*Triggers & wiring*) and the verified hook contract.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from qmx.config import Settings
from qmx.embed import embed_queries, interactive_embedder
from qmx.learnings import INJECT_GLOBAL_SHARE, inject_lessons, render_lesson
from qmx.scope import canonical_repo_key, repo_root
from qmx.store import Store

log = logging.getLogger("qmx.session")

INJECT_CHAR_BUDGET = 10_000  # SessionStart additionalContext hard cap
_PROFILE_FILES = ("CLAUDE.md", "AGENTS.md", "README.md")  # first non-empty one profiles the repo
_PROFILE_CHARS = 2000  # head of that file embedded as the repo profile
_HEADER = "Relevant lessons from past sessions (qmx). Apply them; supersede if now wrong:"


def _payload_cwd(payload: dict) -> str:
    """Working directory for scope resolution across hook sources.

    Claude Code payloads carry ``cwd``; Cursor payloads do not (they expose ``workspace_roots`` on
    stdin and ``CURSOR_PROJECT_DIR`` in the env). Prefer ``cwd``, then the first workspace root,
    then ``CURSOR_PROJECT_DIR``, and finally the process cwd — so both tools scope correctly.
    """
    cwd = payload.get("cwd")
    if cwd:
        return str(cwd)
    roots = payload.get("workspace_roots")
    if isinstance(roots, list) and roots:
        return str(roots[0])
    return os.environ.get("CURSOR_PROJECT_DIR") or str(Path.cwd())


def is_capture_excluded(payload: dict, settings: Settings) -> bool:
    """Whether the payload's workspace folder is (inside) one of ``capture_exclude_roots``.

    Paths are resolved first: Cursor reports symlink-resolved roots (``/tmp`` → ``/private/tmp``).
    """
    if not settings.capture_exclude_roots:
        return False
    cwd = Path(_payload_cwd(payload)).expanduser().resolve()
    return any(
        cwd.is_relative_to(Path(root).expanduser().resolve())
        for root in settings.capture_exclude_roots
    )


def build_injection(
    store: Store,
    scope: str | None,
    *,
    char_budget: int = INJECT_CHAR_BUDGET,
    query_vec: list[float] | None = None,
    global_share: float = INJECT_GLOBAL_SHARE,
) -> str:
    """The ``additionalContext`` block for a scope: header + rendered lessons, or ``""`` if none."""
    body_budget = char_budget - len(_HEADER) - 1
    lessons = inject_lessons(
        store,
        scope,
        char_budget=max(0, body_budget),
        query_vec=query_vec,
        global_share=global_share,
    )
    if not lessons:
        return ""
    lines = "\n".join(render_lesson(le) for le in lessons)
    return f"{_HEADER}\n{lines}"


def repo_profile(cwd: str, scope: str | None) -> str | None:
    """A short description of what the repo is about, used to rank global lessons by relevance.

    The repo key plus the head of the checkout's agent/human overview — ``CLAUDE.md``, else
    ``AGENTS.md``, else ``README.md``. ``None`` outside a git checkout, or when the repo has none
    of these files (the repo key alone is too thin to rank by).
    """
    root = repo_root(cwd)
    if root is None:
        return None
    for name in _PROFILE_FILES:
        try:
            text = (root / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text[:_PROFILE_CHARS].strip()
        if text:
            return f"{scope or root.name}\n{text}"
    return None


def _profile_vector(cwd: str, scope: str | None, settings: Settings) -> list[float] | None:
    """Embed :func:`repo_profile` with the fail-fast read embedder; ``None`` on any failure, so an
    unreachable embedding backend just means the old importance×recency order for global lessons."""
    if not settings.inject_relevance:
        return None
    profile = repo_profile(cwd, scope)
    if profile is None:
        return None
    try:
        with interactive_embedder(settings) as embedder:
            [vec] = embed_queries(embedder, [profile])
        return vec
    except Exception as exc:  # noqa: BLE001 — relevance is a bonus; never block injection
        log.info("session-start relevance skipped: %s", exc)
        return None


def session_start(stdin_text: str, settings: Settings) -> str:
    """Return the JSON a ``SessionStart`` hook should print (``""`` = inject nothing)."""
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
        # Only inject on a fresh start, not resume/compact (avoid re-injecting mid-session).
        if payload.get("source") not in (None, "startup", "clear"):
            return ""
        cwd = _payload_cwd(payload)
        scope = canonical_repo_key(cwd)
        query_vec = _profile_vector(cwd, scope, settings)
        with Store.open(settings.db_path, settings.embed_dim, settings.embed_model) as store:
            context = build_injection(
                store, scope, query_vec=query_vec, global_share=settings.inject_global_share
            )
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
        transcript = payload.get("transcript_path") or os.environ.get("CURSOR_TRANSCRIPT_PATH")
        if not transcript or is_capture_excluded(payload, settings):
            return False
        scope = canonical_repo_key(_payload_cwd(payload))
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
