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
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=settings.request_timeout,
        )

    def complete_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # no thinking tokens — we want the JSON object directly
            "format": schema if schema is not None else "json",
            "options": {"temperature": 0.0},
        }
        content = self._post_with_retry("/api/chat", payload)
        return _parse_json_object(content)

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
