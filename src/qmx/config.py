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
    "QMX_RERANK_MODEL": "rerank_model",
    "QMX_CHAT_MODEL": "chat_model",
    "QMX_MCP_HOST": "mcp_host",
    "QMX_MCP_PORT": "mcp_port",
    "QMX_EMBED_DIM": "embed_dim",
    "QMX_EMBED_BATCH_SIZE": "embed_batch_size",
    "QMX_REQUEST_TIMEOUT": "request_timeout",
    "QMX_MAX_RETRIES": "max_retries",
    "QMX_RETRY_BASE_DELAY": "retry_base_delay",
    "QMX_MEMORY_GLOBS": "memory_globs",
    "QMX_CODE_ROOTS": "code_roots",
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
    embed_model: str = "qwen3-embedding"
    rerank_model: str = "qwen3-reranker"
    chat_model: str = "qwen3"

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
        return d


def _coerce_value(name: str, raw: object) -> object:
    if name == "db_path":
        return Path(raw).expanduser() if not isinstance(raw, Path) else raw
    if name in {"memory_globs", "code_roots"}:
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        return tuple(s.strip() for s in items if str(s).strip())
    if name in {"embed_dim", "embed_batch_size", "max_retries", "mcp_port"}:
        return int(raw)
    if name in {"request_timeout", "retry_base_delay"}:
        return float(raw)
    return str(raw)
