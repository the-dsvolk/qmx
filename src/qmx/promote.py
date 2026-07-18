"""Promotion (Phase E) — graduate a proven learning into per-repo curated memory.

Curated memory is **isolated per GitHub project**: ``qmx promote`` writes a real ``*.md`` (with
valid frontmatter + a ``MEMORY.md`` pointer) into ``<root>/<repo-key>/`` — ``_global/`` for a global
lesson — so repo A's memory never mixes with repo B's, keyed by the canonical repo key
(worktree-stable). Promotion is **human-gated**: it runs only when you approve an eligible lesson
(``qmx lessons --review`` → ``qmx promote <id>``). qmx never auto-edits curated files. See
``plan/qmx-learnings.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

from qmx.store import Learning, Store

DEFAULT_MEMORY_ROOT = "~/.qmx/memory"

# learning.type -> curated memory metadata.type (the memory system's taxonomy).
MEMORY_TYPE_MAP = {"mistake": "feedback", "howto": "feedback", "decision": "project"}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PromotionError(RuntimeError):
    """Raised when a learning cannot be promoted (missing, superseded, already promoted)."""


def repo_dir_name(scope: str | None) -> str:
    """Filesystem-safe per-repo dir: ``owner/repo`` -> ``owner__repo``; ``None`` -> ``_global``."""
    if not scope:
        return "_global"
    return scope.replace("/", "__")


def memory_dir_for(root: Path | str, scope: str | None) -> Path:
    """The curated-memory directory for ``scope`` under ``root`` (created on demand on write)."""
    return Path(root).expanduser() / repo_dir_name(scope)


def slugify(text: str, max_len: int = 60) -> str:
    """Kebab-case slug for a memory filename (stable ``name`` in frontmatter)."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "lesson"


def promotable(store: Store, *, min_importance: float = 0.6, min_reuse: int = 1) -> list[Learning]:
    """Promotion-eligible lessons: live, not yet promoted, importance/reuse over the gate."""
    return store.list_learnings(
        live_only=True,
        exclude_promoted=True,
        min_importance=min_importance,
        min_reuse=min_reuse,
    )


def render_memory(learning: Learning) -> tuple[str, str]:
    """Return ``(slug, file_contents)`` — valid frontmatter + body for the curated memory file."""
    slug = slugify(learning.statement)
    mem_type = MEMORY_TYPE_MAP.get(learning.type, "project")
    detail = learning.detail or "(auto-drafted from a qmx learning)"
    frontmatter = (
        "---\n"
        f"name: {slug}\n"
        f"description: {learning.statement}\n"
        "metadata:\n"
        f"  type: {mem_type}\n"
        f"  qmx_learning_id: {learning.learning_id}\n"
        "---\n"
    )
    if mem_type == "feedback":
        apply = learning.statement
        body = f"{learning.statement}\n\n**Why:** {detail}\n\n**How to apply:** {apply}"
    else:
        body = f"{learning.statement}\n\n{detail}"
    return slug, f"{frontmatter}\n{body}\n"


def promote(
    store: Store, learning_id: int, *, memory_root: Path | str = DEFAULT_MEMORY_ROOT
) -> Path:
    """Write ``learning_id`` into its repo's curated memory and mark it promoted. Returns the path.

    Dedups by slug within the repo dir (updates the matching file instead of duplicating), appends a
    one-line pointer to that dir's ``MEMORY.md``, and records ``promoted_to`` so the lesson stops
    being injected (curated memory now carries it).
    """
    learning = store.get_learning(learning_id)
    if learning is None:
        raise PromotionError(f"no learning #{learning_id}")
    if not learning.is_live:
        raise PromotionError(f"learning #{learning_id} is superseded")

    target_dir = memory_dir_for(memory_root, learning.scope)
    target_dir.mkdir(parents=True, exist_ok=True)
    slug, contents = render_memory(learning)
    path = target_dir / f"{slug}.md"
    path.write_text(contents, encoding="utf-8")  # slug dedup: same lesson updates, not duplicates
    _append_pointer(target_dir / "MEMORY.md", slug, learning)
    store.set_promoted(learning_id, str(path))
    return path


def _append_pointer(index: Path, slug: str, learning: Learning) -> None:
    """Add a one-line pointer to the dir's ``MEMORY.md`` (idempotent on the slug)."""
    line = f"- [{learning.statement}]({slug}.md) — {learning.type}"
    existing = index.read_text(encoding="utf-8") if index.exists() else ""
    if f"]({slug}.md)" in existing:  # already indexed -> refresh the line
        lines = [ln for ln in existing.splitlines() if f"]({slug}.md)" not in ln]
        existing = "\n".join(lines).rstrip()
    header = existing if existing else "# Curated memory (qmx-promoted)"
    index.write_text(f"{header}\n{line}\n", encoding="utf-8")
