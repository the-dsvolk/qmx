"""Chat model — thin judge clients behind a :class:`ChatModel` protocol.

The consolidation model (extract + supersede judgment) is the one qmx component where model quality
matters (see ``plan/qmx-learnings.md``). Like :mod:`qmx.embed`, qmx never loads it in-process — it
POSTs to a model server. :func:`make_chat` picks the client from ``chat_backend``:
:class:`OpenAIChat` ("openai", default) for an OpenAI-compatible server such as vLLM — production
serves ``chat_model = "qwen3.8-judge"`` (Qwen3.8-27B) at ``chat_url`` on the Spark — or
:class:`OllamaChat` ("ollama") for Ollama's ``/api/chat``. The protocol is the seam that lets tests
swap a deterministic fake with no backend.

JSON is schema-constrained (``response_format``/grammar on vLLM, ``format`` on Ollama) so the reply
parses without brittle text scraping; :func:`ChatModel.complete_json` returns the parsed object.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol, runtime_checkable

import httpx

from qmx.config import Settings, template_kwargs_json
from qmx.grammar import UnsupportedSchema, json_grammar

log = logging.getLogger("qmx.chat")

# Consecutive unusable replies after which the judge counts as broken rather than unlucky: a hung or
# misconfigured server would otherwise fail every call "safely" and the run would mark sessions
# consolidated on nothing but fallbacks.
_MAX_UNUSABLE_STREAK = 3
# A cut-off reply ending in at least this much whitespace is the free-whitespace runaway (one
# unescaped `"` closes a JSON string, then the schema grammar admits newlines until the cap). Real
# replies end in "}"; the eval runaways ended in 400-9,600 whitespace characters.
_RUNAWAY_WHITESPACE = 200


class ChatBackendError(RuntimeError):
    """The judge is unreachable or broken (after retries). Callers defer: nothing is marked done."""


def _reply_tail(text: str | None, n: int = 120) -> str:
    """How a cut-off reply ends: its length and last ``n`` non-blank characters, plus the size of
    any trailing whitespace run. Enough to tell a runaway loop (a long whitespace run — the
    unescaped-quote loop of 2026-11-04 — or a repeated phrase) from a genuinely long answer."""
    t = text or ""
    body = t.rstrip()
    ws = len(t) - len(body)
    out = f"{len(t)} chars, ends {body[-n:]!r}"
    return out + (f" + {ws} whitespace chars" if ws >= 20 else "")


class _RunawayReply(Exception):
    """Internal: this reply ran away (whitespace loop or a timeout on a healthy server); the
    client may retry it once with the fixed-layout grammar before reporting a ChatReplyError."""

    def __init__(self, error: ChatReplyError) -> None:
        super().__init__(str(error))
        self.error = error


def _is_whitespace_runaway(text: str | None) -> bool:
    t = text or ""
    return len(t) - len(t.rstrip()) >= _RUNAWAY_WHITESPACE


class ChatReplyError(RuntimeError):
    """The judge is up, but *this one reply* is unusable: cut off at the token cap, no reply before
    the timeout on a healthy server (a runaway generation), or not JSON. Callers fail safe for that
    one call (store as new / block the merge / keep the lesson live) instead of deferring a whole
    run — before 2026-11-04 one looping merge check deferred all of ``qmx consolidate --all``."""


@runtime_checkable
class ChatModel(Protocol):
    """Anything that answers a system+user prompt with a JSON object."""

    def complete_json(
        self, system: str, user: str, schema: dict | None = None, *, max_tokens: int | None = None
    ) -> dict:
        """Return the reply parsed as a JSON object. ``max_tokens`` lowers the output cap for this
        call (short checks); it never raises it above ``chat_max_tokens``. May raise
        :class:`ChatReplyError` (this reply unusable) or :class:`ChatBackendError` (judge down)."""
        ...


class OllamaChat:
    """Retrying client for Ollama's ``/api/chat`` with JSON-constrained output."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._model = settings.chat_model
        self._num_ctx = settings.chat_num_ctx
        self._think = _think_value(settings.chat_think)
        self._max_tokens = settings.chat_max_tokens
        self._temperature = float(settings.chat_temperature)
        self._top_p = float(settings.chat_top_p)
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=settings.chat_timeout,  # judge-only ceiling; see Settings.chat_timeout
        )

    def complete_json(
        self, system: str, user: str, schema: dict | None = None, *, max_tokens: int | None = None
    ) -> dict:
        cap = _effective_cap(self._max_tokens, max_tokens)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self._think,  # Settings.chat_think (default off)
            "format": schema if schema is not None else "json",
            "options": self._options(cap),
        }
        body = self._post_with_retry("/api/chat", payload)
        if body.get("done_reason") == "length":
            content = (body.get("message") or {}).get("content")
            raise ChatReplyError(f"reply cut off at num_predict={cap}; {_reply_tail(content)}")
        return _parse_json_object(body["message"]["content"])

    def _options(self, cap: int) -> dict:
        options: dict = {"temperature": self._temperature}  # Settings.chat_temperature (0.0)
        if self._top_p > 0:  # Settings.chat_top_p; 0 = not sent
            options["top_p"] = self._top_p
        if self._num_ctx > 0:  # 0 = keep the model's own num_ctx
            options["num_ctx"] = self._num_ctx
        if cap > 0:
            options["num_predict"] = cap
        return options

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                body = resp.json()
                body["message"]["content"]  # noqa: B018 — validate the shape before returning
                return body
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
    """Client for an OpenAI-compatible ``/chat/completions`` server (e.g. vLLM).

    Same contract as :class:`OllamaChat`: schema-constrained JSON, temperature ``chat_temperature``
    (default 0), thinking per ``chat_think`` (vLLM/Qwen chat templates take ``enable_thinking`` and
    an optional ``reasoning_effort``), output capped at ``chat_max_tokens``. Off by default, for
    models that need them (Nemotron 3.5 Super): ``chat_top_p`` (sent only when > 0) and
    ``chat_template_kwargs``, whose keys are merged over the computed ``chat_template_kwargs``
    (e.g. ``{"enable_thinking": true, "medium_effort": true}``; Nemotron ignores
    ``reasoning_effort``).

    JSON layout (``chat_json_layout``): "schema" (default) sends ``response_format: json_schema``.
    "indent" (opt-in) sends a per-request grammar admitting exactly ``json.dumps(obj, indent=2)``
    (``structured_outputs.grammar``, see :mod:`qmx.grammar`) — ``response_format``'s free whitespace
    lets a prematurely closed string loop on newlines until the token cap; "indent" falls back to
    "schema" when a server rejects grammars or a schema is outside the subset. Under "schema", a
    reply that runs away (cut off in a long whitespace run, or no reply before the timeout while the
    server is up) is retried **once** with the fixed-layout grammar: no free whitespace, so the loop
    cannot recur, at 3-5x slower decoding paid only on those calls (eval 2026-11-05: every remaining
    cut-off merge check on long lessons was this loop). A genuinely long reply is not retried.

    Failure split: connection errors / 5xx are retried and end in :class:`ChatBackendError`
    (defer). A reply cut off at the cap, a non-JSON reply, or a read timeout while ``/models``
    still answers raise :class:`ChatReplyError` at once — retrying a runaway generation five times
    only burns 25 minutes. Three unusable replies in a row escalate to :class:`ChatBackendError`.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.chat_url:
            raise ValueError('chat_backend = "openai" needs chat_url (e.g. http://host:8000/v1)')
        layout = (settings.chat_json_layout or "indent").strip().lower()
        if layout not in ("indent", "schema"):
            raise ValueError(f'chat_json_layout must be "indent" or "schema", got {layout!r}')
        self._layout = layout
        self._model = settings.chat_model
        self._think = _think_value(settings.chat_think)
        self._temperature = float(settings.chat_temperature)
        self._top_p = float(settings.chat_top_p)
        # Settings.load already validated it; a hand-built Settings may carry anything.
        extra = template_kwargs_json(settings.chat_template_kwargs)
        self._template_kwargs: dict = json.loads(extra) if extra else {}
        self._max_tokens = settings.chat_max_tokens
        self._max_retries = settings.max_retries
        self._base_delay = settings.retry_base_delay
        self._timeout = settings.chat_timeout
        self._url = settings.chat_url.rstrip("/")
        self._unusable_streak = 0
        self._grammar_ok = True  # False once the server rejects per-request grammars
        self.runaway_retries = 0  # fixed-layout retries attempted (for logs/harnesses)
        key = settings.chat_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._url, timeout=settings.chat_timeout, headers=headers
        )

    def complete_json(
        self, system: str, user: str, schema: dict | None = None, *, max_tokens: int | None = None
    ) -> dict:
        cap = _effective_cap(self._max_tokens, max_tokens)
        try:
            try:
                content = self._complete(system, user, schema, cap)
            except _RunawayReply as runaway:
                content = self._retry_fixed_layout(system, user, schema, cap, runaway.error)
            reply = _parse_json_object(content)
            if not reply:
                raise ChatReplyError(f"reply is not a JSON object: {content[:120]!r}")
        except ChatReplyError as exc:
            self._unusable_streak += 1
            if self._unusable_streak >= _MAX_UNUSABLE_STREAK:
                raise ChatBackendError(
                    f"judge at {self._url} returned {self._unusable_streak} unusable replies in a "
                    f"row (last: {exc})"
                ) from exc
            raise
        self._unusable_streak = 0
        return reply

    def _retry_fixed_layout(
        self, system: str, user: str, schema: dict | None, cap: int, error: ChatReplyError
    ) -> str:
        """One retry of a runaway reply under the fixed-layout grammar; else re-raise ``error``."""
        if schema is None or self._layout != "schema" or not self._grammar_ok:
            raise error
        try:
            json_grammar(schema)
        except UnsupportedSchema:
            raise error from None
        self.runaway_retries += 1
        log.warning("%s; retrying once with the fixed-layout grammar", error)
        try:
            content = self._complete(system, user, schema, cap, layout="indent")
        except _RunawayReply as again:
            raise again.error from error
        log.warning("fixed-layout retry succeeded")
        return content

    def _payload(
        self, system: str, user: str, schema: dict | None, cap: int, layout: str | None = None
    ) -> dict:
        # Key order is part of the contract: with default settings the body is byte-identical to
        # what the Qwen3.8 judge was validated with (tests/test_judge_settings.py pins it).
        kwargs: dict = {"enable_thinking": self._think is not False}
        if isinstance(self._think, str):
            kwargs["reasoning_effort"] = self._think
        kwargs.update(self._template_kwargs)  # Settings.chat_template_kwargs: override wins
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
        }
        if self._top_p > 0:
            payload["top_p"] = self._top_p
        payload["chat_template_kwargs"] = kwargs
        if cap > 0:
            payload["max_tokens"] = cap
        if schema is None:
            payload["response_format"] = {"type": "json_object"}
            return payload
        if (layout or self._layout) == "indent":
            try:
                payload["structured_outputs"] = {"grammar": json_grammar(schema)}
                return payload
            except UnsupportedSchema as exc:
                log.debug("fixed-layout grammar unavailable (%s); using response_format", exc)
        payload["response_format"] = {
            "type": "json_schema", "json_schema": {"name": "qmx", "schema": schema}}
        return payload

    def _complete(
        self, system: str, user: str, schema: dict | None, cap: int, layout: str | None = None
    ) -> str:
        """One request (with connection retries). Raises :class:`_RunawayReply` for a runaway,
        :class:`ChatReplyError` for any other unusable reply, :class:`ChatBackendError` when the
        server is unreachable. ``layout`` overrides ``chat_json_layout`` for this call only."""
        payload = self._payload(system, user, schema, cap, layout)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if (resp.status_code == 400 and "structured_outputs" in payload
                        and _mentions_grammar(resp.text)):
                    if layout is not None:  # the fixed-layout retry itself: no second fallback
                        self._grammar_ok = False
                        raise ChatReplyError("fixed-layout retry rejected by the server: "
                                             f"{resp.text[:120]}")
                    # A server without per-request grammars: fall back for the rest of the run.
                    log.warning("judge rejected the fixed-layout grammar (%s); falling back to "
                                "response_format — runaway whitespace loops are possible again",
                                resp.text[:200])
                    self._layout = "schema"
                    payload = self._payload(system, user, schema, cap)
                    resp = self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                content = choice["message"]["content"]
            except httpx.ReadTimeout as exc:
                if self._healthy():
                    raise _RunawayReply(ChatReplyError(
                        f"no reply within {self._timeout:.0f}s although the server is up "
                        "(runaway generation?)"
                    )) from exc
                last_exc = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_exc = exc
            else:
                _log_usage(resp, cap)
                if choice.get("finish_reason") == "length":
                    error = ChatReplyError(
                        f"reply cut off at max_tokens={cap}; {_reply_tail(content)}"
                    )
                    if _is_whitespace_runaway(content):
                        raise _RunawayReply(error)
                    raise error
                return content or ""
            if attempt < self._max_retries - 1:
                time.sleep(self._base_delay * (2**attempt))
        raise ChatBackendError(
            f"judge at {self._url} failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    def _healthy(self) -> bool:
        """Does the server still answer? Tells a runaway generation from a dead/hung judge."""
        try:
            return self._client.get("/models", timeout=10.0).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAIChat:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_chat(settings: Settings) -> OllamaChat | OpenAIChat:
    """The judge client ``settings.chat_backend`` selects ("openai" default, or "ollama")."""
    backend = (settings.chat_backend or "openai").strip().lower()
    if backend == "openai":
        return OpenAIChat(settings)
    if backend != "ollama":
        raise ValueError(
            f'chat_backend must be "ollama" or "openai", got {settings.chat_backend!r}'
        )
    return OllamaChat(settings)


def _effective_cap(configured: int, requested: int | None) -> int:
    """Per-call output cap: ``requested`` may lower ``chat_max_tokens``, never raise it.

    0 means no cap.
    """
    if requested is None or requested <= 0:
        return configured
    return min(configured, requested) if configured > 0 else requested


def _log_usage(resp: httpx.Response, cap: int) -> None:
    """Debug-log the reply's token usage (vLLM ``usage``), so cap headroom can be measured."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    try:
        usage = resp.json().get("usage") or {}
    except ValueError:
        return
    if usage.get("completion_tokens") is not None:
        log.debug("judge reply: %s completion tokens (cap %s), %s prompt tokens",
                  usage.get("completion_tokens"), cap or "none", usage.get("prompt_tokens"))


def _mentions_grammar(text: str) -> bool:
    t = text.lower()
    return "grammar" in t or "structured_outputs" in t or "structured output" in t


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
