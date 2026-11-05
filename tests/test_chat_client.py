"""OllamaChat request shape: judge num_ctx, judge-only timeout, schema JSON with thinking off."""

from __future__ import annotations

import json

import httpx
import pytest

from qmx.chat import OllamaChat
from qmx.config import Settings
from qmx.consolidate import CONSOLIDATE_SCHEMA
from qmx.grammar import json_grammar


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
    assert payload["options"] == {"temperature": 0.0, "num_ctx": 65536, "num_predict": 8192}


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


def test_think_follows_chat_think_setting():
    for raw, expected in (("false", False), ("medium", "medium"), ("true", True), ("", False)):
        captured: list[dict] = []
        OllamaChat(Settings(chat_think=raw), client=_client(captured)).complete_json("s", "u")
        assert captured[0]["think"] == expected, raw


def _openai_client(captured: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path, "auth": request.headers.get("authorization"),
                         "body": json.loads(request.content)})
        reply = {"choices": [{"message": {"content": '{"action": "new"}'}}]}
        return httpx.Response(200, json=reply)

    return httpx.Client(base_url="http://vllm.test/v1", transport=httpx.MockTransport(handler))


def test_openai_backend_request_shape():
    from qmx.chat import OpenAIChat

    captured: list[dict] = []
    s = Settings(chat_backend="openai", chat_url="http://vllm.test/v1", chat_model="qwen3.8-judge",
                 chat_json_layout="indent")
    chat = OpenAIChat(s, client=_openai_client(captured))
    assert chat.complete_json("sys", "user", schema=CONSOLIDATE_SCHEMA) == {"action": "new"}
    [req] = captured
    assert req["path"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "qwen3.8-judge" and body["temperature"] == 0.0
    assert body["max_tokens"] == 8192
    # "indent" layout: a per-request fixed-indent grammar instead of response_format (qmx.grammar).
    assert body["structured_outputs"] == {"grammar": json_grammar(CONSOLIDATE_SCHEMA)}
    assert "response_format" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_backend_effort_and_factory():
    from qmx.chat import OllamaChat as Ollama
    from qmx.chat import OpenAIChat, make_chat

    captured: list[dict] = []
    s = Settings(chat_backend="openai", chat_url="http://vllm.test/v1", chat_think="medium")
    OpenAIChat(s, client=_openai_client(captured)).complete_json("s", "u")
    assert captured[0]["body"]["chat_template_kwargs"] == {
        "enable_thinking": True, "reasoning_effort": "medium"}
    with make_chat(s) as c:
        assert isinstance(c, OpenAIChat)
    with make_chat(Settings(chat_backend="ollama")) as c:
        assert isinstance(c, Ollama)
    with make_chat(Settings()) as c:  # default: the production vLLM judge
        assert isinstance(c, OpenAIChat)


# -- fixed JSON layout, per-call caps, unusable replies vs. a dead judge (2026-11-04) ----------


def _scripted(replies: list, posts: list[dict], healthy: bool = True) -> httpx.Client:
    """vLLM double: each POST pops the next scripted reply (dict body, int status, or exception
    class); GET /models answers 200 when ``healthy``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200 if healthy else 503, json={"data": []})
        posts.append(json.loads(request.content))
        item = replies.pop(0)
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("scripted", request=request)
        if isinstance(item, tuple):
            return httpx.Response(item[0], text=item[1])
        return httpx.Response(200, json=item)

    return httpx.Client(base_url="http://vllm.test/v1", transport=httpx.MockTransport(handler))


def _reply(content: str = '{"action": "new"}', finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def _openai(replies, posts, healthy=True, **kw):
    from qmx.chat import OpenAIChat

    s = Settings(chat_backend="openai", chat_url="http://vllm.test/v1", max_retries=3,
                 retry_base_delay=0.0, **kw)
    return OpenAIChat(s, client=_scripted(replies, posts, healthy))


def test_default_layout_and_schemaless_calls_use_response_format():
    assert Settings().chat_json_layout == "schema"  # grammar layout is opt-in for now
    posts: list[dict] = []
    chat = _openai([_reply(), _reply()], posts)
    chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert posts[0]["response_format"]["json_schema"]["schema"] == CONSOLIDATE_SCHEMA
    assert "structured_outputs" not in posts[0]
    chat.complete_json("s", "u")
    assert posts[1]["response_format"] == {"type": "json_object"}


def test_schema_outside_the_grammar_subset_falls_back_to_response_format():
    posts: list[dict] = []
    odd = {"type": "object", "properties": {"a": {"type": "string", "pattern": "x"}},
           "required": ["a"]}
    _openai([_reply('{"a": "x"}')], posts, chat_json_layout="indent").complete_json(
        "s", "u", schema=odd)
    assert "structured_outputs" not in posts[0] and posts[0]["response_format"]["json_schema"]


def test_per_call_cap_can_lower_but_not_raise_chat_max_tokens():
    posts: list[dict] = []
    chat = _openai([_reply(), _reply()], posts, chat_max_tokens=4000)
    chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA, max_tokens=3072)
    chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA, max_tokens=9999)
    assert [p["max_tokens"] for p in posts] == [3072, 4000]


def test_reply_cut_off_at_the_cap_is_unusable_and_not_retried():
    from qmx.chat import ChatReplyError

    posts: list[dict] = []
    chat = _openai([_reply('{"reasoning": "the core ', finish="length")], posts)
    with pytest.raises(ChatReplyError, match="cut off"):
        chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert len(posts) == 1


def test_non_json_reply_is_unusable():
    from qmx.chat import ChatReplyError

    with pytest.raises(ChatReplyError, match="not a JSON object"):
        _openai([_reply("sorry, no")], []).complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)


def test_read_timeout_on_a_healthy_server_is_a_runaway_not_an_outage():
    from qmx.chat import ChatReplyError

    posts: list[dict] = []
    chat = _openai([httpx.ReadTimeout], posts, healthy=True)
    with pytest.raises(ChatReplyError, match="runaway"):
        chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert len(posts) == 1  # was: five 300 s retries, then the whole run deferred


def test_read_timeout_on_a_dead_server_is_retried_then_defers():
    from qmx.chat import ChatBackendError

    posts: list[dict] = []
    chat = _openai([httpx.ReadTimeout] * 3, posts, healthy=False)
    with pytest.raises(ChatBackendError):
        chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert len(posts) == 3


def test_connection_errors_still_defer_after_retries():
    from qmx.chat import ChatBackendError

    posts: list[dict] = []
    with pytest.raises(ChatBackendError):
        _openai([(503, "busy")] * 3, posts).complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert len(posts) == 3


def test_three_unusable_replies_in_a_row_escalate_to_backend_error():
    from qmx.chat import ChatBackendError, ChatReplyError

    cut = _reply("{", finish="length")
    chat = _openai([cut, cut, _reply(), cut, cut, cut], [])
    for _ in range(2):
        with pytest.raises(ChatReplyError):
            chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA) == {"action": "new"}  # resets
    for _ in range(2):
        with pytest.raises(ChatReplyError):
            chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    with pytest.raises(ChatBackendError, match="3 unusable replies in a row"):
        chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)


def test_server_rejecting_grammars_falls_back_for_the_rest_of_the_run():
    posts: list[dict] = []
    chat = _openai([(400, "Invalid grammar: unknown rule"), _reply(), _reply()], posts,
                   chat_json_layout="indent")
    assert chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA) == {"action": "new"}
    chat.complete_json("s", "u", schema=CONSOLIDATE_SCHEMA)
    assert "structured_outputs" in posts[0]
    assert all("response_format" in p for p in posts[1:])


def test_invalid_json_layout_setting_is_rejected():
    from qmx.chat import OpenAIChat

    with pytest.raises(ValueError, match="chat_json_layout"):
        OpenAIChat(Settings(chat_backend="openai", chat_url="http://x/v1", chat_json_layout="x"))


def test_ollama_per_call_cap_and_cut_off_reply():
    from qmx.chat import ChatReplyError

    captured: list[dict] = []
    OllamaChat(Settings(), client=_client(captured)).complete_json("s", "u", max_tokens=3072)
    assert captured[0]["options"]["num_predict"] == 3072

    def cut(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"a'}, "done_reason": "length"})

    client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(cut))
    with pytest.raises(ChatReplyError):
        OllamaChat(Settings(), client=client).complete_json("s", "u")
