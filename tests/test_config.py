"""Config resolution: defaults < TOML file < QMX_* env vars, with type coercion."""

from __future__ import annotations

from pathlib import Path

from qmx.config import Settings


def test_defaults():
    s = Settings.load(config_path=Path("/does/not/exist"), env={})
    assert s.ollama_url == "http://localhost:11434"
    assert s.embed_model == "qwen3-embedding:0.6b"
    assert s.embed_dim == 1024


def test_env_overrides_and_coerces():
    env = {
        "QMX_OLLAMA_URL": "http://spark-0e81.local:11434",
        "QMX_EMBED_DIM": "2560",
        "QMX_REQUEST_TIMEOUT": "12.5",
        "QMX_DB_PATH": "/tmp/qmx/index.db",
    }
    s = Settings.load(config_path=Path("/does/not/exist"), env=env)
    assert s.ollama_url == "http://spark-0e81.local:11434"
    assert s.embed_dim == 2560 and isinstance(s.embed_dim, int)
    assert s.request_timeout == 12.5 and isinstance(s.request_timeout, float)
    assert s.db_path == Path("/tmp/qmx/index.db")


def test_toml_then_env_precedence(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'ollama_url = "http://from-toml:11434"\nembed_dim = 512\nextra_section = "ignored"\n'
    )
    # TOML alone
    s = Settings.load(config_path=cfg, env={})
    assert s.ollama_url == "http://from-toml:11434"
    assert s.embed_dim == 512
    # env wins over TOML
    s2 = Settings.load(config_path=cfg, env={"QMX_EMBED_DIM": "999"})
    assert s2.embed_dim == 999
    assert s2.ollama_url == "http://from-toml:11434"


def test_as_dict_is_json_safe():
    s = Settings.load(config_path=Path("/does/not/exist"), env={})
    d = s.as_dict()
    assert isinstance(d["db_path"], str)
