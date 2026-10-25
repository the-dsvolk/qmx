"""capture_exclude_roots: capture / session-end ignore sessions whose workspace is excluded."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from qmx import capture as capture_mod
from qmx import session as session_mod
from qmx.config import Settings
from qmx.session import is_capture_excluded


@pytest.fixture
def roots(tmp_path):
    excluded = tmp_path / "memoir"
    call = excluded / "2026-10-22_sim-t4-migration"
    call.mkdir(parents=True)
    other = tmp_path / "work" / "repo"
    other.mkdir(parents=True)
    sibling = tmp_path / "memoir-dev"  # shares the prefix string, not the folder
    sibling.mkdir()
    return excluded, call, other, sibling


def _settings(tmp_path, *roots: str) -> Settings:
    return Settings(db_path=tmp_path / "qmx.db", capture_exclude_roots=tuple(roots))


def test_setting_from_toml_and_env(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('capture_exclude_roots = ["~/memoir"]\n')
    assert Settings.load(cfg, env={}).capture_exclude_roots == ("~/memoir",)
    env = {"QMX_CAPTURE_EXCLUDE_ROOTS": "~/memoir, /tmp/x"}
    assert Settings.load(cfg, env=env).capture_exclude_roots == ("~/memoir", "/tmp/x")
    assert Settings.load(cfg, env={}).as_dict()["capture_exclude_roots"] == ["~/memoir"]


def test_is_capture_excluded_matches_root_and_children_only(tmp_path, roots, monkeypatch):
    excluded, call, other, sibling = roots
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    settings = _settings(tmp_path, str(excluded))
    assert is_capture_excluded({"workspace_roots": [str(call)]}, settings)
    assert is_capture_excluded({"cwd": str(excluded)}, settings)
    assert not is_capture_excluded({"workspace_roots": [str(other)]}, settings)
    assert not is_capture_excluded({"cwd": str(sibling)}, settings)
    assert not is_capture_excluded({"cwd": str(call)}, _settings(tmp_path))


def test_is_capture_excluded_resolves_symlinks_and_tilde(tmp_path, roots, monkeypatch):
    excluded, call, _, _ = roots
    link = tmp_path / "link"
    link.symlink_to(excluded)
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = _settings(tmp_path, "~/link")
    assert is_capture_excluded({"workspace_roots": [str(call)]}, settings)


def test_is_capture_excluded_falls_back_to_cursor_project_dir(tmp_path, roots, monkeypatch):
    excluded, call, _, _ = roots
    monkeypatch.setenv("CURSOR_PROJECT_DIR", str(call))
    assert is_capture_excluded({"cwd": None}, _settings(tmp_path, str(excluded)))


class _StoreSpy:
    opened = 0

    @classmethod
    def open(cls, *args, **kwargs):
        cls.opened += 1
        raise RuntimeError("store opened")


def test_capture_skips_excluded_workspace(tmp_path, roots, monkeypatch):
    excluded, call, other, _ = roots
    monkeypatch.setattr(capture_mod, "Store", _StoreSpy)
    _StoreSpy.opened = 0
    settings = _settings(tmp_path, str(excluded))
    transcript = str(tmp_path / "t.jsonl")

    payload = {"transcript_path": transcript, "workspace_roots": [str(call)]}
    assert capture_mod.capture(json.dumps(payload), settings, source="cursor") == 0
    assert _StoreSpy.opened == 0

    payload = {"transcript_path": transcript, "workspace_roots": [str(other)]}
    assert capture_mod.capture(json.dumps(payload), settings, source="cursor") == 0
    assert _StoreSpy.opened == 1


def test_session_end_does_not_spawn_for_excluded_workspace(tmp_path, roots, monkeypatch):
    excluded, call, other, _ = roots
    spawned: list[list[str]] = []
    fake = SimpleNamespace(Popen=lambda cmd, **kw: spawned.append(cmd), DEVNULL=subprocess.DEVNULL)
    monkeypatch.setattr(session_mod, "subprocess", fake)
    settings = _settings(tmp_path, str(excluded))
    transcript = str(tmp_path / "t.jsonl")

    payload = {"transcript_path": transcript, "workspace_roots": [str(call)]}
    assert session_mod.session_end(json.dumps(payload), settings) is False
    assert spawned == []

    payload = {"transcript_path": transcript, "workspace_roots": [str(other)]}
    assert session_mod.session_end(json.dumps(payload), settings) is True
    assert len(spawned) == 1 and spawned[0][3:6] == ["consolidate", "--session", transcript]
