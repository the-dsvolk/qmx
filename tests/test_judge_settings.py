"""Judge client settings for running other models from config alone (2026-11-05).

``chat_temperature``, ``chat_top_p``, ``chat_template_kwargs``, ``chat_short_max_tokens``,
``judge_reasoning_field`` and ``judge_filing_rules`` are all off by default: with defaults the
request body must stay byte-identical to what the Qwen3.8 judge was validated with. The Super
profile below is the one in INFRA.md (PROPOSAL A.5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from qmx.chat import OllamaChat, OpenAIChat
from qmx.config import Settings
from qmx.consolidate import CONSOLIDATE_SCHEMA, SHORT_REPLY_MAX_TOKENS

SUPER_PROFILE = """
chat_backend = "openai"
chat_url = "http://spark-b.test:8001/v1"
chat_model = "nemotron-3.5-super-vl"
chat_think = "true"
chat_template_kwargs = { enable_thinking = true, medium_effort = true }
chat_temperature = 1.0
chat_top_p = 0.95
chat_max_tokens = 32768
chat_short_max_tokens = 16384
chat_timeout = 1200
chat_json_layout = "schema"
judge_reasoning_field = false
judge_filing_rules = "v2"
"""

NO_CONFIG = Path("/does/not/exist")


# -- settings: defaults, env, TOML, coercion ------------------------------------------------------


def test_new_settings_default_to_todays_behaviour():
    s = Settings.load(config_path=NO_CONFIG, env={})
    assert s.chat_temperature == 0.0 and isinstance(s.chat_temperature, float)
    assert s.chat_top_p == 0.0
    assert s.chat_template_kwargs == ""
    assert s.chat_short_max_tokens == SHORT_REPLY_MAX_TOKENS == 3072
    assert s.judge_reasoning_field is True
    assert s.judge_filing_rules == "current"
    assert Settings() == s  # dataclass defaults == resolved defaults


def test_new_settings_from_env_are_coerced():
    env = {
        "QMX_CHAT_TEMPERATURE": "1",
        "QMX_CHAT_TOP_P": "0.95",
        "QMX_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": true,  "medium_effort": true}',
        "QMX_CHAT_SHORT_MAX_TOKENS": "16384",
        "QMX_JUDGE_REASONING_FIELD": "off",
        "QMX_JUDGE_FILING_RULES": "v2",
    }
    s = Settings.load(config_path=NO_CONFIG, env=env)
    assert s.chat_temperature == 1.0 and isinstance(s.chat_temperature, float)
    assert s.chat_top_p == 0.95
    assert s.chat_template_kwargs == '{"enable_thinking": true, "medium_effort": true}'
    assert s.chat_short_max_tokens == 16384 and isinstance(s.chat_short_max_tokens, int)
    assert s.judge_reasoning_field is False
    assert s.judge_filing_rules == "v2"


def test_super_profile_from_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(SUPER_PROFILE)
    s = Settings.load(config_path=cfg, env={})
    # The TOML inline table is stored as a canonical JSON string (Settings is frozen + slotted).
    assert s.chat_template_kwargs == '{"enable_thinking": true, "medium_effort": true}'
    assert (s.chat_temperature, s.chat_top_p) == (1.0, 0.95)
    assert (s.chat_max_tokens, s.chat_short_max_tokens, s.chat_timeout) == (32768, 16384, 1200.0)
    assert (s.judge_reasoning_field, s.judge_filing_rules) == (False, "v2")
    json.dumps(s.as_dict())  # still JSON-safe for `qmx status`
    # env still wins over TOML
    s2 = Settings.load(config_path=cfg, env={"QMX_JUDGE_REASONING_FIELD": "1"})
    assert s2.judge_reasoning_field is True


@pytest.mark.parametrize(("raw", "expected"), [
    ("true", True), ("1", True), ("yes", True), ("on", True), (" TRUE ", True), (True, True),
    ("false", False), ("0", False), ("no", False), ("Off", False), (False, False), (0, False),
])
def test_reasoning_field_bool_coercion(raw, expected):
    assert Settings._coerce({"judge_reasoning_field": raw}).judge_reasoning_field is expected


def test_reasoning_field_rejects_anything_else():
    with pytest.raises(ValueError, match="judge_reasoning_field"):
        Settings._coerce({"judge_reasoning_field": "maybe"})


@pytest.mark.parametrize(("raw", "expected"), [
    ("", ""), ("   ", ""), ({}, "{}"), ({"a": 1, "b": True}, '{"a": 1, "b": true}'),
    (' {"a":1,"b":true} ', '{"a": 1, "b": true}'),
])
def test_template_kwargs_canonical_string(raw, expected):
    assert Settings._coerce({"chat_template_kwargs": raw}).chat_template_kwargs == expected


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"medium"', "3", "null"])
def test_template_kwargs_must_be_a_json_object(bad):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        Settings._coerce({"chat_template_kwargs": bad})


# -- OpenAI payload ------------------------------------------------------------------------------


def _vllm(posts: list[bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.content)
        choice = {"message": {"content": '{"action": "new"}'}, "finish_reason": "stop"}
        reply = {"choices": [choice], "usage": {"prompt_tokens": 900, "completion_tokens": 412}}
        return httpx.Response(200, json=reply)

    return httpx.Client(base_url="http://vllm.test/v1", transport=httpx.MockTransport(handler))


def _openai(posts: list[bytes], **kw) -> OpenAIChat:
    s = Settings(chat_backend="openai", chat_url="http://vllm.test/v1", **kw)
    return OpenAIChat(s, client=_vllm(posts))


def _as_sent(body: dict) -> bytes:
    """The exact bytes httpx puts on the wire for ``body``."""
    return httpx.Request("POST", "http://vllm.test/v1/chat/completions", json=body).content


def test_default_payload_is_byte_identical_to_the_validated_qwen_request():
    posts: list[bytes] = []
    _openai(posts, chat_model="qwen3.8-judge").complete_json(
        "sys", "user", schema=CONSOLIDATE_SCHEMA, max_tokens=SHORT_REPLY_MAX_TOKENS)
    expected = {  # main @ 98bfd69, OpenAIChat._payload with chat_json_layout = "schema"
        "model": "qwen3.8-judge",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 3072,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "qmx", "schema": CONSOLIDATE_SCHEMA}},
    }
    [raw] = posts
    assert raw == _as_sent(expected)  # keys, order and number formatting ("0.0", not "0")
    body = json.loads(raw)
    assert list(body) == list(expected) and "top_p" not in body


def test_super_profile_payload(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(SUPER_PROFILE)
    posts: list[bytes] = []
    chat = OpenAIChat(Settings.load(config_path=cfg, env={}), client=_vllm(posts))
    chat.complete_json("sys", "user", schema=CONSOLIDATE_SCHEMA, max_tokens=16384)
    chat.complete_json("sys", "turns")  # an extract-style call: chat_max_tokens
    body = json.loads(posts[0])
    assert list(body) == ["model", "messages", "temperature", "top_p", "chat_template_kwargs",
                          "max_tokens", "response_format"]
    assert body["model"] == "nemotron-3.5-super-vl"
    assert (body["temperature"], body["top_p"]) == (1.0, 0.95)
    # chat_think = "true" -> enable_thinking; the override adds Nemotron's medium level.
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "medium_effort": True}
    assert body["max_tokens"] == 16384
    assert json.loads(posts[1])["max_tokens"] == 32768
    assert chat._timeout == 1200.0  # noqa: SLF001 — the httpx timeout of a real client


def test_template_kwargs_override_wins_over_computed_ones():
    posts: list[bytes] = []
    _openai(posts, chat_think="medium",
            chat_template_kwargs='{"enable_thinking": false, "reasoning_effort": "low", "x": 1}',
            ).complete_json("s", "u")
    assert json.loads(posts[0])["chat_template_kwargs"] == {
        "enable_thinking": False, "reasoning_effort": "low", "x": 1}


def test_top_p_zero_is_not_sent_and_temperature_is_a_float():
    posts: list[bytes] = []
    _openai(posts, chat_temperature=1, chat_top_p=0.0).complete_json("s", "u")
    body = json.loads(posts[0])
    assert "top_p" not in body
    assert body["temperature"] == 1.0 and b'"temperature":1.0' in posts[0]


def test_invalid_template_kwargs_rejected_at_construction():
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        OpenAIChat(Settings(chat_backend="openai", chat_url="http://x/v1",
                            chat_template_kwargs="{bad json"))


def test_completion_tokens_are_logged_at_debug(caplog):
    posts: list[bytes] = []
    with caplog.at_level(logging.DEBUG, logger="qmx.chat"):
        _openai(posts).complete_json("s", "u", max_tokens=3072)
    assert "412 completion tokens (cap 3072)" in caplog.text


# -- Ollama options --------------------------------------------------------------------------------


def _ollama(captured: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"action": "new"}'}})

    return httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_options_default_and_overrides():
    captured: list[dict] = []
    OllamaChat(Settings(), client=_ollama(captured)).complete_json("s", "u")
    assert list(captured[0]["options"].items()) == [
        ("temperature", 0.0), ("num_ctx", 65536), ("num_predict", 8192)]
    OllamaChat(Settings(chat_temperature=1.0, chat_top_p=0.95),
               client=_ollama(captured)).complete_json("s", "u", max_tokens=3072)
    assert list(captured[1]["options"].items()) == [
        ("temperature", 1.0), ("top_p", 0.95), ("num_ctx", 65536), ("num_predict", 3072)]
