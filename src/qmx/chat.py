"""Chat model — a thin Ollama ``/api/chat`` client behind a :class:`ChatModel` protocol.

The consolidation model (extract + supersede judgment) is the one qmx component where model quality
matters (see ``plan/qmx-learnings.md``). Like :mod:`qmx.embed`, qmx never loads it in-process — it
POSTs to Ollama (``chat_model``, e.g. ``qwen3.6:35b-a3b``, served on the Spark). The protocol is the
seam that lets tests swap a deterministic fake with no backend.

JSON is requested via Ollama's ``format`` field (schema-constrained decoding) so the reply parses
without brittle text scraping; :func:`ChatModel.complete_json` returns the parsed object.
"""

from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

import httpx

from qmx.config import Settings


class ChatBackendError(RuntimeError):
    """Raised when the chat backend is unreachable/invalid after all retries."""


@runtime_checkable
class ChatModel(Protocol):
    """Anything that answers a system+user prompt with a JSON object."""

    def complete_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        """Return the model's reply parsed as a JSON object (``{}`` on an empty/garbage reply)."""
        ...


class OllamaChat:
    """Retrying client for Ollama's ``/api/chat`` with JSON-constrained output."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._model = settings.chat_model
        self._num_ctx = settings.chat_num_ctx
        self._think = _think_value(settings.chat_think)
        self._max_tokens = settings.chat_max_tokens
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=settings.chat_timeout,  # judge-only ceiling; see Settings.chat_timeout
        )

    def complete_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self._think,  # Settings.chat_think (default off)
            "format": schema if schema is not None else "json",
            "options": self._options(),
        }
        content = self._post_with_retry("/api/chat", payload)
        return _parse_json_object(content)

    def _options(self) -> dict:
        options: dict = {"temperature": 0.0}
        if self._num_ctx > 0:  # 0 = keep the model's own num_ctx
            options["num_ctx"] = self._num_ctx
        if self._max_tokens > 0:
            options["num_predict"] = self._max_tokens
        return options

    def _post_with_retry(self, path: str, payload: dict) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (httpx.HTTPError, KeyError) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._base_delay * (2**attempt))
        raise ChatBackendError(
            f"Ollama chat failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaChat:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OpenAIChat:
    """Retrying client for an OpenAI-compatible ``/chat/completions`` server (e.g. vLLM).

    Same contract as :class:`OllamaChat`: schema-constrained JSON (``response_format``),
    temperature 0, thinking per ``chat_think`` (vLLM/Qwen chat templates take ``enable_thinking``
    and an optional ``reasoning_effort``), output capped at ``chat_max_tokens``.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.chat_url:
            raise ValueError('chat_backend = "openai" needs chat_url (e.g. http://host:8000/v1)')
        self._model = settings.chat_model
        self._think = _think_value(settings.chat_think)
        self._max_tokens = settings.chat_max_tokens
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._url = settings.chat_url.rstrip("/")
        key = settings.chat_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._url, timeout=settings.chat_timeout, headers=headers
        )

    def complete_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        kwargs: dict = {"enable_thinking": self._think is not False}
        if isinstance(self._think, str):
            kwargs["reasoning_effort"] = self._think
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "response_format": (
                {"type": "json_schema", "json_schema": {"name": "qmx", "schema": schema}}
                if schema is not None else {"type": "json_object"}
            ),
            "chat_template_kwargs": kwargs,
        }
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                return _parse_json_object(resp.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._base_delay * (2**attempt))
        raise ChatBackendError(
            f"judge at {self._url} failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAIChat:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_chat(settings: Settings) -> OllamaChat | OpenAIChat:
    """The judge client ``settings.chat_backend`` selects ("ollama" default, or "openai")."""
    backend = (settings.chat_backend or "ollama").strip().lower()
    if backend == "openai":
        return OpenAIChat(settings)
    if backend != "ollama":
        raise ValueError(
            f'chat_backend must be "ollama" or "openai", got {settings.chat_backend!r}'
        )
    return OllamaChat(settings)


def _think_value(raw: str) -> bool | str:
    """Map ``chat_think`` to Ollama's ``think``: False / True / an effort-level string."""
    v = (raw or "").strip().lower()
    if v in ("", "false", "off", "no", "0"):
        return False
    if v in ("true", "on", "yes", "1"):
        return True
    return v


def _parse_json_object(content: str) -> dict:
    """Parse a JSON object from a model reply, tolerating stray prose around it."""
    content = (content or "").strip()
    if not content:
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            value = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}
