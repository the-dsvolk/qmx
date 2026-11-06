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
from collections.abc import Iterable
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


# Spellings that mean "global" when someone types a scope (``None`` is the real global scope).
GLOBAL_SCOPE_ALIASES = frozenset({"", "global", "_global", "none", "null"})


class UnknownScopeError(ValueError):
    """A bare scope name that matches no known ``owner/repo`` key (or matches several)."""


def normalize_scope(
    raw: str | None, known: Iterable[str] = (), *, strict: bool = False
) -> str | None:
    """The canonical scope key for a scope someone typed — ``None`` (global) or ``owner/repo``.

    Session hooks derive scope from the git remote, so their keys are already canonical. A scope
    passed by hand to ``add_learning``/``update_learning`` is not: ``"global"`` was stored as a
    literal (matching no session at all) and ``"qmx"`` never matches ``"the-dsvolk/qmx"`` sessions.
    So, against ``known`` (the scope keys already in use):

    - ``None``, ``""``, ``"global"``, ``"none"``, ``"null"`` → ``None``;
    - a git URL or path → its ``owner/repo`` (:func:`normalize_remote_url`);
    - a known key in any letter case → that key's stored spelling;
    - any other ``owner/repo`` → accepted as is (a repo with no lessons yet);
    - a bare name → the one known ``owner/repo`` whose repo part it is (``"qmx"`` →
      ``"the-dsvolk/qmx"``), else a known bare key spelled the same (a repo with no remote).

    A bare name that stays unresolved — no match, or several ``owner/repo`` matches — raises
    :class:`UnknownScopeError` when ``strict`` (a person or agent typed it and can fix it), and is
    kept as given otherwise (automated callers must never fail on it).
    """
    if raw is None:
        return None
    value = raw.strip()
    if value.lower() in GLOBAL_SCOPE_ALIASES:
        return None
    if "://" in value or value.startswith("git@") or value.endswith(".git") or value.count("/") > 1:
        value = normalize_remote_url(value) or value
    keys = {k for k in known if k}
    by_lower = {k.lower(): k for k in keys if "/" in k}
    if value.lower() in by_lower:
        return by_lower[value.lower()]
    if "/" in value:
        return value
    owners = sorted(k for k in by_lower.values() if k.rsplit("/", 1)[1].lower() == value.lower())
    if len(owners) == 1:
        return owners[0]
    bare = {k.lower(): k for k in keys if "/" not in k}
    if not owners and value.lower() in bare:
        return bare[value.lower()]
    if strict:
        if owners:
            raise UnknownScopeError(f"scope {value!r} is ambiguous: use one of {', '.join(owners)}")
        raise UnknownScopeError(
            f"unknown scope {value!r}: pass the repo's git remote as owner/repo "
            f"(e.g. the-dsvolk/qmx), or omit it / pass 'global' for a global lesson"
        )
    return value


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
