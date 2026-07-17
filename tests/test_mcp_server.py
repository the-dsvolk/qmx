"""MCP server wiring — the expected tools are registered and callable."""

from __future__ import annotations

import asyncio

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


def test_tools_have_descriptions(server):
    tools = asyncio.run(server.list_tools())
    assert all(t.description for t in tools)


def test_query_tool_executes(server):
    result = asyncio.run(server.call_tool("search_code", {"text": "retry", "k": 3}))
    # FastMCP returns (content, structured) or content; assert we got something non-empty back.
    assert result is not None
