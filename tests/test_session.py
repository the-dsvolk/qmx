"""Phase D: cwd→git-remote scope normalization + SessionStart injection payload."""

from __future__ import annotations

import json

import pytest

from qmx.config import Settings
from qmx.learnings import add_learning
from qmx.scope import canonical_repo_key, normalize_remote_url
from qmx.session import _payload_cwd, build_injection, session_start
from qmx.store import Store
from tests.fakes import FakeEmbedder


@pytest.mark.parametrize(
    "url,expected",
    [
        ("git@github.com:the-dsvolk/qmx.git", "the-dsvolk/qmx"),
        ("https://github.com/the-dsvolk/qmx.git", "the-dsvolk/qmx"),
        ("https://github.com/Cruise/xtorch", "Cruise/xtorch"),
        ("ssh://git@example.com:22/org/team/repo.git", "team/repo"),
        ("", None),
    ],
)
def test_normalize_remote_url(url, expected):
    assert normalize_remote_url(url) == expected


def test_canonical_repo_key_uses_remote_not_dirname(tmp_path):
    import subprocess

    repo = tmp_path / "weird-worktree-name"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:the-dsvolk/qmx.git"],
        cwd=repo,
        check=True,
    )
    # Keyed by the remote, not the directory basename.
    assert canonical_repo_key(repo) == "the-dsvolk/qmx"


def test_canonical_repo_key_none_outside_git(tmp_path):
    assert canonical_repo_key(tmp_path) is None


def test_payload_cwd_prefers_claude_cwd_then_cursor_sources(monkeypatch):
    # Claude: explicit cwd wins.
    assert _payload_cwd({"cwd": "/claude/repo"}) == "/claude/repo"
    # Cursor: no cwd -> first workspace_root.
    monkeypatch.setenv("CURSOR_PROJECT_DIR", "/env/proj")
    assert _payload_cwd({"workspace_roots": ["/ws/root", "/ws/other"]}) == "/ws/root"
    # Cursor: no cwd, no roots -> CURSOR_PROJECT_DIR env.
    assert _payload_cwd({}) == "/env/proj"


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def test_build_injection_scopes_and_excludes_other_repos(store):
    s, embedder = store
    add_learning(s, embedder, type="mistake", statement="A repo lesson", scope="me/repoA")
    add_learning(s, embedder, type="decision", statement="global lesson", scope=None)
    add_learning(s, embedder, type="mistake", statement="B repo lesson", scope="me/repoB")
    text = build_injection(s, "me/repoA")
    assert "A repo lesson" in text
    assert "global lesson" in text
    assert "B repo lesson" not in text


def test_build_injection_empty_when_no_lessons(store):
    s, _ = store
    assert build_injection(s, "me/repoA") == ""


def test_session_start_emits_additional_context(tmp_path):
    embedder = FakeEmbedder(dim=64)
    db = tmp_path / "index.db"
    with Store.open(db, embedder.dim, "fake") as s:
        add_learning(s, embedder, type="howto", statement="use uv not pip", scope=None)
    settings = Settings(
        db_path=db, embed_dim=64, embed_model="fake", ollama_url="http://127.0.0.1:9"
    )
    # No cwd -> global lessons still inject (source=startup).
    out = session_start(json.dumps({"source": "startup", "cwd": str(tmp_path)}), settings)
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "use uv not pip" in payload["hookSpecificOutput"]["additionalContext"]


def test_session_start_skips_on_resume(tmp_path):
    embedder = FakeEmbedder(dim=64)
    db = tmp_path / "index.db"
    with Store.open(db, embedder.dim, "fake") as s:
        add_learning(s, embedder, type="howto", statement="use uv not pip", scope=None)
    settings = Settings(db_path=db, embed_dim=64, embed_model="fake")
    assert session_start(json.dumps({"source": "resume"}), settings) == ""
