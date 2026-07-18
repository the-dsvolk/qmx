"""Scope resolution — map a working directory to a stable, per-repo key.

Injection relevance is *project identity*, not meaning (there is no query at ``SessionStart``), so a
lesson's ``scope`` is keyed to the repo. The key is the **git remote** normalized to ``owner/repo``,
not the directory name — so every worktree/clone of one repo shares a single coherent scope (a
worktree path like ``.claude/worktrees/qmx-x`` would otherwise fragment it). See
``plan/qmx-learnings.md`` (*Relevance & scope*).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_SSH_RE = re.compile(r"^[^@]+@[^:]+:(?P<path>.+)$")  # git@github.com:owner/repo(.git)


def normalize_remote_url(url: str) -> str | None:
    """Normalize a git remote URL to a canonical ``owner/repo`` key (host-independent)."""
    url = (url or "").strip()
    if not url:
        return None
    m = _SSH_RE.match(url)
    path = m.group("path") if m else re.sub(r"^[a-z]+://[^/]+/", "", url)
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else None


def canonical_repo_key(cwd: Path | str) -> str | None:
    """The ``owner/repo`` key for the repo containing ``cwd``, or ``None`` if it is not a git repo.

    Prefers ``origin``'s normalized URL (worktree-stable); falls back to the repo root's basename.
    """
    cwd = str(cwd)
    remote = _git(cwd, "remote", "get-url", "origin")
    if remote:
        key = normalize_remote_url(remote)
        if key:
            return key
    top = _git(cwd, "rev-parse", "--show-toplevel")
    return Path(top).name if top else None


def _git(cwd: str, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None
