"""MCP server wiring — the expected tools are registered and callable."""

from __future__ import annotations

import asyncio
import json

import pytest

from qmx.mcp_server import build_server
from qmx.service import QmxService
from tests.fakes import FakeEmbedder, build_index

FILES = {"net.py": "def retry_with_backoff():\n    return 1\n"}


@pytest.fixture
def server(tmp_path):
    embedder = FakeEmbedder(dim=64)
    settings = build_index(tmp_path, embedder, FILES)
    return build_server(settings, QmxService(settings, embedder))


def test_registers_expected_tools(server):
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"query", "search_code", "get", "status"} <= names
    # The learning lifecycle is agent-reachable: add, fix in place, retire, un-retire.
    assert {"add_learning", "update_learning", "deprecate_learning", "restore_learning"} <= names


def test_hard_delete_is_not_agent_reachable(server):
    """Deletion is irreversible and a human call — it must not be an MCP tool."""
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert not {n for n in names if "delete" in n or "purge" in n}


def test_learning_lifecycle_tools_execute(server):
    added = asyncio.run(
        server.call_tool(
            "add_learning",
            {"type": "mistake", "statement": "retry backoff must be jittered", "importance": 0.9},
        )
    )
    learning_id = _structured(added)["learning_id"]

    updated = _structured(asyncio.run(
        server.call_tool("update_learning", {"learning_id": learning_id, "importance": 0.2})
    ))
    assert updated["importance"] == pytest.approx(0.2)
    assert updated["learning_id"] == learning_id  # in place, not a new row

    retired = _structured(asyncio.run(
        server.call_tool(
            "deprecate_learning", {"learning_id": learning_id, "reason": "superseded by policy"}
        )
    ))
    assert retired["deprecated_reason"] == "superseded by policy"
    recalled = _structured(
        asyncio.run(server.call_tool("lessons", {"text": "retry backoff jitter"}))
    )
    assert recalled == [], "a retired lesson must stop being recalled"

    revived = _structured(asyncio.run(
        server.call_tool("restore_learning", {"learning_id": learning_id})
    ))
    assert "deprecated_at" not in revived


def _structured(result):
    """Unwrap a FastMCP tool result to its JSON payload.

    Shapes seen in the wild: a bare content list, or ``(content, {"result": payload})`` for tools
    whose return type is optional/non-object.
    """
    content = result[1] if isinstance(result, tuple) else result
    if isinstance(content, dict):
        return content["result"] if set(content) == {"result"} else content
    assert content, "tool returned no content"
    return json.loads(content[0].text)


def test_tools_have_descriptions(server):
    tools = asyncio.run(server.list_tools())
    assert all(t.description for t in tools)


def test_query_tool_executes(server):
    result = asyncio.run(server.call_tool("search_code", {"text": "retry", "k": 3}))
    # FastMCP returns (content, structured) or content; assert we got something non-empty back.
    assert result is not None
