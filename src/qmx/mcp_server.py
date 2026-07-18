"""Resident MCP server — the primary way Claude Code talks to qmx.

Exposes the read side (``query`` / ``search_code`` / ``get`` / ``status``) as ``mcp__qmx__*`` tools
over an HTTP endpoint so one server on the Spark serves every Claude Code instance on the LAN
(``plan/qmx-deployment.md``). The write door (chat capture) arrives in Phase 4.
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
        """Semantic + keyword search over the qmx knowledge base (code today; chats later).

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
    def lessons(text: str, k: int = 5, type: str | None = None) -> list[dict]:
        """Recall distilled **lessons** — decisions, mistakes+corrections, how-tos learned before.

        Higher-signal than raw chat recall: ranked by relevance × importance × recency, each with
        citations. ``type`` filters ``decision`` | ``mistake`` | ``howto``.
        """
        return svc.lessons(text, k=k, type=type)

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
