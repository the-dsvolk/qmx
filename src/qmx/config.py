"""Configuration — the per-machine seam that lets the same code run on the Mac (dev), the DGX
Spark (prod), and in CI.

Resolution order (lowest to highest precedence): dataclass defaults → optional TOML file
(``$QMX_CONFIG`` or ``~/.qmx/config.toml``) → ``QMX_*`` environment variables. See
``plan/qmx-deployment.md`` for how ``QMX_OLLAMA_URL`` points the Mac at the Spark.
"""

from __future__ import annotations

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
}

# Where Claude memory lives. Globs (``~`` expanded) matching memory dirs or .md files; a dir match
# is scanned recursively for *.md. Indexed as ``kind="memory"``.
DEFAULT_MEMORY_GLOBS = ("~/.claude/projects/*/memory",)


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved qmx settings. Immutable; build one with :meth:`load` at startup."""

    # Model backend (Ollama). On the Mac this points at the Spark; on the Spark, localhost.
    ollama_url: str = "http://localhost:11434"

    # Where the single, rebuildable index lives (on the Spark in prod).
    db_path: Path = DEFAULT_DB_PATH

    # Qwen model tags (as served by Ollama).
    embed_model: str = "qwen3-embedding:0.6b"
    rerank_model: str = "qwen3-reranker"
    # Consolidation/learnings judge — read from here, never hardcoded (see plan/qmx-learnings.md).
    chat_model: str = "qwen3.6:35b-a3b"
    # Judge context window, sent per request as Ollama ``num_ctx`` (the served qwen3.6 build pins
    # 32768 in its Modelfile). On an oversized prompt Ollama silently keeps only about half the
    # window, so at 32K long sessions were judged on 26-46% of their turns; 65536 fixes that at the
    # same speed and memory (hybrid attention: 23 GB at both). 0 = leave it to the model default.
    chat_num_ctx: int = 65536
    # Judge HTTP timeout. Deliberately separate from ``request_timeout``: that one is also the read
    # timeout of the synchronous Stop-hook embed and of MCP queries, which must stay short. The
    # judge runs detached at session end, so a generous ceiling blocks nobody.
    chat_timeout: float = 300.0
    # Judge thinking mode, sent as Ollama ``think``: "false" (default — the reasoning-first JSON
    # field gives most of the benefit at a fraction of the cost), "true", or an effort level
    # ("low"/"medium"/"high") for models that support them. qwen3.6 has no levels: any non-false
    # value is full thinking (~5-6 min/session in the 2026-11 eval).
    chat_think: str = "false"
    # Where the judge runs. "ollama" (default) uses ``ollama_url``'s /api/chat; "openai" uses an
    # OpenAI-compatible server (e.g. vLLM) at ``chat_url`` (base URL incl. /v1, e.g.
    # "http://spark-0f50.local:8000/v1"), with optional bearer ``chat_api_key``.
    chat_backend: str = "ollama"
    chat_url: str = ""
    chat_api_key: str = ""
    # Cap on judge output tokens (reasoning field included) so one rambling reply cannot run away.
    chat_max_tokens: int = 8192
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

    # Embedding dimensionality — must match the served model; stored in ``meta`` so a mismatch
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


def _coerce_value(name: str, raw: object) -> object:
    if name == "db_path":
        return Path(raw).expanduser() if not isinstance(raw, Path) else raw
    if name in {"memory_globs", "code_roots", "capture_exclude_roots"}:
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        return tuple(s.strip() for s in items if str(s).strip())
    if name in {
        "embed_dim", "embed_batch_size", "max_retries", "mcp_port", "parked_stale_days",
        "chat_num_ctx", "consolidate_min_turns", "consolidate_min_chars", "chat_max_tokens",
    }:
        return int(raw)
    if name in {"request_timeout", "retry_base_delay", "chat_timeout", "dedup_min_similarity"}:
        return float(raw)
    return str(raw)
