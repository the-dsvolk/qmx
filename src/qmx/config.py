"""Configuration — the per-machine seam that lets the same code run on the Mac (dev), the DGX
Spark (prod), and in CI.

Resolution order (lowest to highest precedence): dataclass defaults → optional TOML file
(``$QMX_CONFIG`` or ``~/.qmx/config.toml``) → ``QMX_*`` environment variables. See
``plan/qmx-deployment.md`` for how ``QMX_OLLAMA_URL`` points the Mac at the Spark.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".qmx" / "index.db"
DEFAULT_CONFIG_PATH = Path.home() / ".qmx" / "config.toml"

# Env var name -> Settings field name. Only these keys are read from the environment.
_ENV_MAP = {
    "QMX_OLLAMA_URL": "ollama_url",
    "QMX_DB_PATH": "db_path",
    "QMX_EMBED_MODEL": "embed_model",
    "QMX_EMBED_BACKEND": "embed_backend",
    "QMX_EMBED_URL": "embed_url",
    "QMX_EMBED_QUERY_PREFIX": "embed_query_prefix",
    "QMX_EMBED_PASSAGE_PREFIX": "embed_passage_prefix",
    "QMX_EMBED_MAX_TOKENS": "embed_max_tokens",
    "QMX_RERANK_URL": "rerank_url",
    "QMX_RERANK_MODEL": "rerank_model",
    "QMX_CHAT_MODEL": "chat_model",
    "QMX_CHAT_NUM_CTX": "chat_num_ctx",
    "QMX_CHAT_TIMEOUT": "chat_timeout",
    "QMX_CHAT_THINK": "chat_think",
    "QMX_CHAT_BACKEND": "chat_backend",
    "QMX_CHAT_URL": "chat_url",
    "QMX_CHAT_API_KEY": "chat_api_key",
    "QMX_CHAT_MAX_TOKENS": "chat_max_tokens",
    "QMX_CHAT_JSON_LAYOUT": "chat_json_layout",
    "QMX_CHAT_TEMPERATURE": "chat_temperature",
    "QMX_CHAT_TOP_P": "chat_top_p",
    "QMX_CHAT_TEMPLATE_KWARGS": "chat_template_kwargs",
    "QMX_CHAT_SHORT_MAX_TOKENS": "chat_short_max_tokens",
    "QMX_JUDGE_REASONING_FIELD": "judge_reasoning_field",
    "QMX_JUDGE_FILING_RULES": "judge_filing_rules",
    "QMX_JUDGE_OFFRULE_BLOCK": "judge_offrule_block",
    "QMX_DEDUP_MIN_SIMILARITY": "dedup_min_similarity",
    "QMX_CONSOLIDATE_MIN_TURNS": "consolidate_min_turns",
    "QMX_CONSOLIDATE_MIN_CHARS": "consolidate_min_chars",
    "QMX_MCP_HOST": "mcp_host",
    "QMX_MCP_PORT": "mcp_port",
    "QMX_EMBED_DIM": "embed_dim",
    "QMX_EMBED_BATCH_SIZE": "embed_batch_size",
    "QMX_REQUEST_TIMEOUT": "request_timeout",
    "QMX_MAX_RETRIES": "max_retries",
    "QMX_RETRY_BASE_DELAY": "retry_base_delay",
    "QMX_MEMORY_GLOBS": "memory_globs",
    "QMX_CODE_ROOTS": "code_roots",
    "QMX_CAPTURE_EXCLUDE_ROOTS": "capture_exclude_roots",
    "QMX_PROMOTED_MEMORY_ROOT": "promoted_memory_root",
    "QMX_PARKED_ROOT": "parked_root",
    "QMX_PARKED_STALE_DAYS": "parked_stale_days",
    "QMX_PARKED_SIMILAR_MAX_DISTANCE": "parked_similar_max_distance",
}

# Where Claude memory lives. Globs (``~`` expanded) matching memory dirs or .md files; a dir match
# is scanned recursively for *.md. Indexed as ``kind="memory"``.
DEFAULT_MEMORY_GLOBS = ("~/.claude/projects/*/memory",)


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved qmx settings. Immutable; build one with :meth:`load` at startup."""

    # Model backend (Ollama). On the Mac this points at the Spark; on the Spark, localhost.
    ollama_url: str = "http://localhost:11434"

    # Embedding backend. "ollama" (default) posts to ``ollama_url``'s /api/embed. "openai" posts to
    # an OpenAI-compatible /v1/embeddings server (vLLM) at ``embed_url`` (base URL incl. /v1), e.g.
    # Nemotron-3-Embed-1B: "http://spark-0f50.local:8002/v1". Changing the model means re-embedding
    # the index (``qmx reembed``); ``meta`` refuses a DB built with a different model or dim.
    embed_backend: str = "ollama"
    embed_url: str = ""
    # Asymmetric retrieval prefixes, prepended client-side (qmx.embed). ``embed_query_prefix`` goes
    # on search queries; ``embed_passage_prefix`` on everything stored and on the doc-vs-doc checks
    # (insert-time dedup, similar parked items). Empty (default) = text sent as-is, which is what
    # the Qwen index was built with. Nemotron Embed needs "query: " / "passage: ".
    embed_query_prefix: str = ""
    embed_passage_prefix: str = ""
    # "openai" backend only: vLLM ``truncate_prompt_tokens``, so an over-long chunk is cut to this
    # many tokens server-side instead of rejected. 0 = not sent.
    embed_max_tokens: int = 0

    # Where the single, rebuildable index lives (on the Spark in prod).
    db_path: Path = DEFAULT_DB_PATH

    # Qwen model tags (as served by Ollama).
    embed_model: str = "qwen3-embedding:0.6b"
    rerank_model: str = "qwen3-reranker"
    # Consolidation/learnings judge — read from here, never hardcoded (see plan/qmx-learnings.md).
    # Default: the production judge, Qwen3.8-27B NVFP4 + DFlash2 served by vLLM under this name.
    chat_model: str = "qwen3.8-judge"
    # Judge context window for the "ollama" backend only, sent per request as Ollama ``num_ctx``
    # (vLLM fixes it server-side with --max-model-len). On an oversized prompt Ollama silently
    # keeps only about half the window, so at 32K long sessions were judged on 26-46% of their
    # turns (the retired qwen3.6 judge pinned 32768); 65536 fixed that. 0 = the model default.
    chat_num_ctx: int = 65536
    # Judge HTTP timeout. Deliberately separate from ``request_timeout``: that one is also the read
    # timeout of the synchronous Stop-hook embed and of MCP queries, which must stay short. The
    # judge runs detached at session end, so a generous ceiling blocks nobody.
    chat_timeout: float = 300.0
    # Judge thinking mode: "false" (default — the reasoning-first JSON field gives most of the
    # benefit at a fraction of the cost), "true", or an effort level ("low"/"medium"/"high") for
    # models that support them. Sent as ``enable_thinking``/``reasoning_effort`` chat-template
    # kwargs ("openai" backend) or as Ollama ``think`` ("ollama" backend).
    chat_think: str = "false"
    # Where the judge runs. "openai" (default) uses an OpenAI-compatible server (vLLM) at
    # ``chat_url`` (base URL incl. /v1; default: the production judge on the primary spark), with
    # optional bearer ``chat_api_key``; "ollama" uses ``ollama_url``'s /api/chat. Override
    # ``chat_url`` in config when the judge lives elsewhere.
    chat_backend: str = "openai"
    chat_url: str = "http://spark-0e81.local:8000/v1"
    chat_api_key: str = ""
    # Cap on judge output tokens (reasoning field included) so one rambling reply cannot run away.
    # Short judge calls (decisions and checks) use a lower per-call cap; see qmx.consolidate.
    chat_max_tokens: int = 8192
    # JSON layout the "openai" backend asks for. "schema" (default): response_format json_schema.
    # "indent": a per-request grammar admitting exactly json.dumps(indent=2) — no free whitespace,
    # so a string the model closes early cannot loop on newlines (qmx.grammar). Opt-in for now:
    # on 2026-11-04 the generated grammar decoded 3-5x slower than xgrammar's own EBNF for the same
    # language (likely the missing string lookahead), so it stays off until that is fixed. Loops
    # under "schema" are bounded by the per-call caps and fail safe (ChatReplyError).
    chat_json_layout: str = "schema"
    # Judge sampling and chat-template overrides — all off by default, so the request body stays
    # exactly what the Qwen3.8 judge was validated with (temperature 0, no top_p, thinking per
    # ``chat_think``). A model that needs other settings (e.g. Nemotron 3.5 Super: temperature 1.0,
    # top_p 0.95, ``{"enable_thinking": true, "medium_effort": true}``) runs from config alone.
    # ``chat_top_p`` 0 = not sent. ``chat_template_kwargs`` is a JSON object (a TOML inline table or
    # a JSON string; stored as a canonical JSON string so Settings stays immutable); its keys are
    # merged over the ``enable_thinking``/``reasoning_effort`` that ``chat_think`` computes
    # ("openai" backend only). See INFRA.md for the Super profile.
    chat_temperature: float = 0.0
    chat_top_p: float = 0.0
    chat_template_kwargs: str = ""
    # Output cap for the short judge calls (decision, merge check, retirement check, dedup check);
    # still capped by ``chat_max_tokens``. A thinking model needs more: thinking shares the budget.
    chat_short_max_tokens: int = 3072
    # Judge prompt wrapper. true (default): every judge schema starts with a required `reasoning`
    # string and the prompts tell the model to work in it — essential for Qwen3.8 with thinking off
    # (without it, empty lesson lists). false: field and wording removed from all five calls, for a
    # model that thinks natively (Super with medium thinking).
    judge_reasoning_field: bool = True
    # Judge filing rules: "current" (default) or "v2" — main-rule comparison, a code-computed main
    # rule (first sentence) in every check, a deterministic headline backstop on merges, an
    # off-rule-additions check in the merge check and a two-quote retirement check
    # (qmx.consolidate).
    judge_filing_rules: str = "current"
    # v2 only. false (default): a merge the judge flags as adding a different rule is applied and
    # logged (the headline backstop still guards the old main rule; blocking turned good merges into
    # near-duplicates in validation). true: it is refused and the candidate stored as new.
    judge_offrule_block: bool = False
    # One-off gate for ``qmx consolidate`` / session-end: sessions with fewer turns or characters
    # than this are skipped (and left un-consolidated so a resumed session can accumulate).
    consolidate_min_turns: int = 8
    consolidate_min_chars: int = 4000
    # Insert-time near-duplicate check: a `new` lesson whose nearest live lesson is at least this
    # similar gets one "same lesson?" question (drop / merge / keep). 0 disables it.
    dedup_min_similarity: float = 0.65

    # Reranker: base URL of a Cohere-style /v1/rerank server (llama.cpp llama-server --reranking).
    # Empty = disabled (RRF-only). e.g. "http://spark-0e81.local:8081".
    rerank_url: str = ""

    # Resident MCP server bind address (the Spark serves this on the LAN; clients use QMX_MCP_URL).
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765

    # Embedding dimensionality — must match the served model (or, for the "openai" backend, be at
    # most its width: wider vectors are cut and re-normalised); stored in ``meta`` so a mismatch
    # forces a rebuild rather than silently corrupting the vector table.
    embed_dim: int = 1024

    # HTTP client behaviour.
    embed_batch_size: int = 64
    request_timeout: float = 60.0
    max_retries: int = 5
    retry_base_delay: float = 0.5

    # Claude memory sources (kind="memory"). A TOML list, or comma-separated in the env var.
    memory_globs: tuple[str, ...] = DEFAULT_MEMORY_GLOBS

    # Code repos to keep indexed (kind="code"), swept by ``qmx refresh``. Same list format.
    code_roots: tuple[str, ...] = ()

    # Workspaces whose agent sessions the capture/session-end hooks ignore (``~`` expanded; a
    # session is skipped when its workspace folder is one of these or inside one). Same list format.
    capture_exclude_roots: tuple[str, ...] = ()

    # Root of the per-repo curated-memory store that ``qmx promote`` writes to (isolated by repo
    # key: ``<root>/<owner>__<repo>/``, ``<root>/_global/``). See ``plan/qmx-learnings.md``.
    promoted_memory_root: str = "~/.qmx/memory"

    # Parked items ("park this, come back later") — one markdown file each, the source of truth;
    # closed ones move to ``<root>/archive/``. Open items untouched for ``parked_stale_days`` are
    # flagged for a keep-or-drop review (never auto-archived). See ``plan/qmx-parked.md``.
    parked_root: str = "~/.qmx/parked"
    parked_stale_days: int = 30
    # Cosine distance under which another open parked item counts as "similar" (a dedup hint on
    # park). Model-specific: recalibrate when the embedding model changes.
    parked_similar_max_distance: float = 0.3

    @classmethod
    def load(cls, config_path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
        """Build settings from defaults, an optional TOML file, then ``QMX_*`` env vars."""
        env = os.environ if env is None else env
        values: dict[str, object] = {}

        path = config_path or Path(env.get("QMX_CONFIG", DEFAULT_CONFIG_PATH))
        if path.is_file():
            with path.open("rb") as fh:
                values.update(tomllib.load(fh))

        for env_key, field_name in _ENV_MAP.items():
            if env_key in env:
                values[field_name] = env[env_key]

        return cls._coerce(values)

    @classmethod
    def _coerce(cls, values: dict[str, object]) -> Settings:
        """Coerce raw string/TOML values to the field types, ignoring unknown keys."""
        types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, object] = {}
        for name, raw in values.items():
            if name not in types:
                continue  # ignore stray keys so config files can carry extra sections
            kwargs[name] = _coerce_value(name, raw)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["db_path"] = str(self.db_path)
        d["memory_globs"] = list(self.memory_globs)
        d["code_roots"] = list(self.code_roots)
        d["capture_exclude_roots"] = list(self.capture_exclude_roots)
        return d


_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def _coerce_value(name: str, raw: object) -> object:
    if name == "db_path":
        return Path(raw).expanduser() if not isinstance(raw, Path) else raw
    if name in {"memory_globs", "code_roots", "capture_exclude_roots"}:
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        return tuple(s.strip() for s in items if str(s).strip())
    if name in {
        "embed_dim", "embed_batch_size", "max_retries", "mcp_port", "parked_stale_days",
        "chat_num_ctx", "consolidate_min_turns", "consolidate_min_chars", "chat_max_tokens",
        "chat_short_max_tokens", "embed_max_tokens",
    }:
        return int(raw)
    if name in {
        "request_timeout", "retry_base_delay", "chat_timeout", "dedup_min_similarity",
        "chat_temperature", "chat_top_p", "parked_similar_max_distance",
    }:
        return float(raw)
    if name in {"judge_reasoning_field", "judge_offrule_block"}:
        return _coerce_bool(name, raw)
    if name == "chat_template_kwargs":
        return template_kwargs_json(raw)
    return str(raw)


def _coerce_bool(name: str, raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    v = str(raw).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false (or 1/0, yes/no, on/off), got {raw!r}")


def template_kwargs_json(raw: object) -> str:
    """Canonical JSON string for ``chat_template_kwargs``; "" when unset.

    Accepts a mapping (a TOML inline table) or a string holding a JSON object; anything else —
    including invalid JSON or a JSON value that is not an object — raises ``ValueError``.
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return json.dumps(raw)
    text = str(raw).strip()
    if not text:
        return ""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"chat_template_kwargs is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"chat_template_kwargs must be a JSON object, got {text!r}")
    return json.dumps(value)
