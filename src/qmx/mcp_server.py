"""Resident MCP server — the primary way Claude Code talks to qmx.

Exposes the tools ``query`` / ``search_code`` / ``recall`` / ``lessons`` / ``add_learning`` /
``update_learning`` / ``deprecate_learning`` / ``restore_learning`` / ``park`` / ``list_parked`` /
``unpark`` / ``resume_parked`` / ``get`` / ``status`` as
``mcp__qmx__*`` over an HTTP endpoint, so one server on the Spark serves every Claude Code instance
on the LAN (``plan/qmx-deployment.md``). Chat capture is a separate write path (the ``qmx capture``
Stop hook).

The write tools are the learning lifecycle: add, fix in place, soft-retire, un-retire. Hard delete
is deliberately **not** part of this surface — it is irreversible and its two legitimate uses
(flat-out wrong with no historical value; a lesson that captured something confidential) are human
calls, so it is planned as a CLI-only verb rather than an agent-reachable tool.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from qmx.config import Settings
from qmx.service import QmxService


def build_server(settings: Settings, service: QmxService | None = None) -> FastMCP:
    """Build the FastMCP server with qmx's tools bound to a :class:`QmxService`."""
    svc = service if service is not None else QmxService(settings)
    server = FastMCP("qmx", host=settings.mcp_host, port=settings.mcp_port)

    @server.tool()
    def query(text: str, k: int = 5, kind: str | None = None) -> list[dict]:
        """Semantic + keyword search over the qmx knowledge base (code, docs, chats, learnings).

        Returns ranked hits with ``path``, ``start_line``/``end_line``, ``symbol``, ``score`` and a
        text snippet. Optional ``kind`` filters to ``code`` | ``doc`` | ``chat`` | ``learning``.
        """
        return svc.query(text, k=k, kind=kind)

    @server.tool()
    def search_code(text: str, k: int = 5) -> list[dict]:
        """Search only code by meaning; returns ``file:line`` locations with snippets."""
        return svc.query(text, k=k, kind="code")

    @server.tool()
    def recall(text: str, k: int = 5) -> list[dict]:
        """Recall past Claude Code conversations — semantic search over indexed chat turns.

        Returns matching turns with their transcript path, line, and role (user/assistant).
        """
        return svc.recall(text, k=k)

    @server.tool()
    def lessons(
        text: str, k: int = 5, type: str | None = None, include_retired: bool = False
    ) -> list[dict]:
        """Recall distilled **lessons** — decisions, mistakes+corrections, how-tos learned before.

        Higher-signal than raw chat recall: ranked by relevance × importance × recency, each with
        citations. ``type`` filters ``decision`` | ``mistake`` | ``howto``. ``include_retired`` also
        returns deprecated/superseded lessons (hidden by default) with their ``deprecated_reason``
        and ``superseded_by`` breadcrumb — use it to check *why* something was retired.
        """
        return svc.lessons(text, k=k, type=type, include_retired=include_retired)

    @server.tool()
    def add_learning(
        type: str,
        statement: str,
        detail: str | None = None,
        topic: str | None = None,
        scope: str | None = None,
        importance: float = 0.5,
    ) -> dict:
        """Record a durable lesson so future sessions recall it. ``type``: decision|mistake|howto.

        ``statement`` is the lesson in one crisp sentence; ``detail`` the why/correction/better-way;
        ``scope`` the repo key it applies to (omit for a global lesson).
        """
        return svc.add_learning(
            type=type,
            statement=statement,
            detail=detail,
            topic=topic,
            scope=scope,
            importance=importance,
        )

    @server.tool()
    def update_learning(
        learning_id: int,
        statement: str | None = None,
        detail: str | None = None,
        topic: str | None = None,
        type: str | None = None,
        scope: str | None = None,
        importance: float | None = None,
    ) -> dict | None:
        """Fix an existing lesson **in place** — correct its wording or lower its ``importance``.

        Prefer this over adding a near-duplicate when a lesson is merely wrong, stale in detail, or
        over-weighted; use ``deprecate_learning`` when it should stop firing altogether. Omitted
        fields are left unchanged (to *clear* one: ``qmx update-learning --clear-detail``).
        Returns the updated lesson, or ``None`` if ``learning_id`` does not exist.
        """
        patch = {
            key: value
            for key, value in (
                ("statement", statement),
                ("detail", detail),
                ("topic", topic),
                ("type", type),
                ("scope", scope),
                ("importance", importance),
            )
            if value is not None
        }
        return svc.update_learning(learning_id, **patch)

    @server.tool()
    def deprecate_learning(
        learning_id: int, reason: str | None = None, superseded_by: int | None = None
    ) -> dict | None:
        """Soft-retire a lesson: it stops being recalled, but is kept with a breadcrumb.

        The right tool when a lesson turned out to be wrong or went stale — reversible, and
        preferable to leaving it firing at a lower weight. ``reason`` records why; ``superseded_by``
        points at the lesson that replaces it (omit when nothing does). Retired lessons are visible
        again via ``lessons(..., include_retired=True)`` and revivable with ``restore_learning``.
        """
        return svc.deprecate_learning(learning_id, reason=reason, superseded_by=superseded_by)

    @server.tool()
    def restore_learning(learning_id: int) -> dict | None:
        """Un-retire a lesson deprecated by mistake — clears the retire/supersede markers."""
        return svc.restore_learning(learning_id)

    @server.tool()
    def park(
        text: str,
        why: str | None = None,
        cwd: str | None = None,
        scope: str | None = None,
        session: str | None = None,
    ) -> dict:
        """Park an idea to come back to later — when the user says "park this", "remind me later",
        "let's come back to that", or mid-task surfaces a side idea not worth derailing for.

        ``text`` is the idea in one line; ``why`` the context needed to pick it up cold (what
        prompted it, what to check). Pass ``cwd`` (your working directory) so the item is scoped
        to that repo, or ``scope`` (``owner/repo``) directly; omit both for a global item.
        ``session`` optionally records the transcript it came from. Returns the stored item;
        ``similar`` lists open items that look alike (mention them — it may be a duplicate).
        """
        return svc.park(text, why=why, scope=scope, cwd=cwd, session=session)

    @server.tool()
    def list_parked(
        cwd: str | None = None,
        scope: str | None = None,
        all_scopes: bool = False,
        status: str = "open",
        query: str | None = None,
        stale_only: bool = False,
        k: int = 20,
    ) -> list[dict]:
        """List parked items — only call when the user asks ("what did I park?", "parked items").

        Scoped to ``cwd``'s repo (or ``scope``) plus global items; with neither, every scope.
        ``status``: ``open`` (default) | ``done`` | ``dropped`` | ``closed`` | ``all``. ``query``
        ranks by meaning. ``stale_only`` keeps items untouched long enough to need a keep-or-drop
        review (each item also carries a ``stale`` flag).
        """
        return svc.list_parked(
            scope=scope,
            cwd=cwd,
            all_scopes=all_scopes,
            status=status,
            query=query,
            stale_only=stale_only,
            k=k,
        )

    @server.tool()
    def unpark(item_id: int, status: str = "done", note: str | None = None) -> dict:
        """Close a parked item once the work is done (``done``) or abandoned (``dropped``).

        ``note`` records the resolution (what was done, or why it was dropped). The item is
        archived, not deleted — ``qmx reopen <id>`` brings it back.
        """
        return svc.unpark(item_id, status=status, note=note)

    @server.tool()
    def resume_parked(item_id: int, k: int = 3) -> dict:
        """Pick a parked item back up: the item plus the past chat turns that best match it,
        so the original context is restored, not just the one-line note."""
        return svc.resume_parked(item_id, k=k)

    @server.tool()
    def get(chunk_id: int) -> dict | None:
        """Fetch a single chunk's full text + location by ``chunk_id`` (from a prior result)."""
        return svc.get(chunk_id)

    @server.tool()
    def status() -> dict:
        """Index stats (documents/chunks/mentions) and Ollama backend health."""
        return svc.status()

    return server


def serve(
    settings: Settings,
    *,
    transport: str = "streamable-http",
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the MCP server (blocking). ``transport`` is ``streamable-http`` or ``stdio``."""
    server = build_server(settings)
    if host is not None:
        server.settings.host = host
    if port is not None:
        server.settings.port = port
    server.run(transport=transport)
