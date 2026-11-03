"""OllamaChat request shape: judge num_ctx, judge-only timeout, schema JSON with thinking off."""

from __future__ import annotations

import json

import httpx

from qmx.chat import OllamaChat
from qmx.config import Settings
from qmx.consolidate import CONSOLIDATE_SCHEMA


def _client(captured: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"action": "new"}'}})

    return httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_request_sends_judge_num_ctx_and_qmx_defaults():
    captured: list[dict] = []
    chat = OllamaChat(Settings(chat_model="judge:test"), client=_client(captured))
    assert chat.complete_json("sys", "user", schema=CONSOLIDATE_SCHEMA) == {"action": "new"}
    [payload] = captured
    assert payload["model"] == "judge:test"
    assert payload["think"] is False
    assert payload["format"] == CONSOLIDATE_SCHEMA
    assert payload["options"] == {"temperature": 0.0, "num_ctx": 65536}


def test_num_ctx_zero_leaves_the_model_default():
    captured: list[dict] = []
    chat = OllamaChat(Settings(chat_num_ctx=0), client=_client(captured))
    chat.complete_json("sys", "user")
    assert "num_ctx" not in captured[0]["options"]


def test_judge_uses_chat_timeout_not_request_timeout():
    # request_timeout also bounds the synchronous Stop-hook embed + MCP queries; it must not be
    # what the slow, detached judge waits on (and vice versa).
    with OllamaChat(Settings(chat_timeout=300.0, request_timeout=60.0)) as chat:
        assert chat._client.timeout.read == 300.0  # noqa: SLF001
