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


# --- SessionStart relevance: repo lessons first, global lessons ranked by the repo profile -------


@pytest.fixture
def wide_store():
    embedder = FakeEmbedder(dim=256)  # wider hash space: fewer accidental word collisions
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _statements(text: str) -> list[str]:
    return [line.split("] ", 1)[1].split(" — ")[0] for line in text.splitlines()[1:]]


def test_inject_puts_repo_lessons_before_global(wide_store):
    s, embedder = wide_store
    add_learning(s, embedder, type="howto", statement="global high", importance=1.0)
    add_learning(s, embedder, type="howto", statement="repo low", scope="me/tax", importance=0.2)
    assert _statements(build_injection(s, "me/tax")) == ["repo low", "global high"]


def test_inject_ranks_global_lessons_by_relevance_not_importance(wide_store):
    s, embedder = wide_store
    add_learning(
        s,
        embedder,
        type="howto",
        importance=1.0,
        statement="Size GPU commitments to the reservation floor of cluster nodes",
    )
    add_learning(
        s,
        embedder,
        type="howto",
        importance=0.6,
        statement="Track wash sale freezes across household brokerage accounts and tax lots",
    )
    [vec] = embedder.embed(["portfolio tax lots wash sale brokerage household accounts"])
    ranked = _statements(build_injection(s, "me/tax", query_vec=vec))
    assert ranked[0].startswith("Track wash sale")
    # Without a profile vector the old importance order still applies.
    assert _statements(build_injection(s, "me/tax"))[0].startswith("Size GPU")


def test_inject_reserves_a_share_for_global_lessons(wide_store):
    s, embedder = wide_store
    for i in range(30):
        add_learning(s, embedder, type="howto", scope="me/tax", statement=f"repo lesson {i} " * 8)
    add_learning(s, embedder, type="howto", statement="the one global lesson " + "x" * 140)
    text = build_injection(s, "me/tax", char_budget=3000, global_share=0.2)
    assert "the one global lesson" in text
    assert len(text) <= 3000
    # share 0: the repo's lessons take the whole budget, globals only fill unused space.
    assert "the one global lesson" not in build_injection(
        s, "me/tax", char_budget=3000, global_share=0.0
    )


def test_inject_outside_a_repo_never_shows_other_repos_lessons(wide_store):
    s, embedder = wide_store
    add_learning(s, embedder, type="howto", statement="repo only", scope="me/tax")
    add_learning(s, embedder, type="howto", statement="everyone")
    assert _statements(build_injection(s, None)) == ["everyone"]


def test_inject_skips_a_too_long_lesson_instead_of_stopping(wide_store):
    s, embedder = wide_store
    add_learning(s, embedder, type="howto", statement="huge " * 400, importance=1.0)
    add_learning(s, embedder, type="howto", statement="short one", importance=0.5)
    assert _statements(build_injection(s, None, char_budget=600)) == ["short one"]


def _git_repo(path, files: dict[str, str]):
    import subprocess

    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    for name, text in files.items():
        (path / name).write_text(text)
    return path


def test_repo_profile_prefers_claude_md_then_readme(tmp_path):
    from qmx.session import repo_profile

    both = _git_repo(tmp_path / "both", {"CLAUDE.md": "agent notes", "README.md": "human notes"})
    assert repo_profile(str(both), "me/both") == "me/both\nagent notes"
    readme = _git_repo(tmp_path / "readme", {"README.md": "human notes"})
    assert repo_profile(str(readme), None) == "readme\nhuman notes"
    assert repo_profile(str(_git_repo(tmp_path / "bare", {})), "me/bare") is None
    assert repo_profile(str(tmp_path), None) is None  # not a git checkout


def _settings(db, **kw):
    return Settings(db_path=db, embed_dim=256, embed_model="fake", **kw)


def _seed_tax_vs_gpu(db):
    embedder = FakeEmbedder(dim=256)
    with Store.open(db, embedder.dim, "fake") as s:
        add_learning(
            s,
            embedder,
            type="howto",
            importance=1.0,
            statement="Size GPU commitments to the reservation floor of cluster nodes",
        )
        add_learning(
            s,
            embedder,
            type="howto",
            importance=0.6,
            statement="Track wash sale freezes across household brokerage accounts and tax lots",
        )
    return embedder


def _first_injected(out: str) -> str:
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    return _statements(context)[0]


def test_session_start_ranks_globals_by_the_repo_profile(tmp_path, monkeypatch):
    import contextlib

    db = tmp_path / "index.db"
    embedder = _seed_tax_vs_gpu(db)
    repo = _git_repo(
        tmp_path / "ledger",
        {"CLAUDE.md": "Household portfolio: tax lots, wash sale tracking, brokerage accounts."},
    )
    monkeypatch.setattr(
        "qmx.session.interactive_embedder", lambda _s: contextlib.nullcontext(embedder)
    )
    out = session_start(json.dumps({"source": "startup", "cwd": str(repo)}), _settings(db))
    assert _first_injected(out).startswith("Track wash sale")
    # The switch turns relevance off: back to importance order, and no embed call is made.
    monkeypatch.setattr("qmx.session.interactive_embedder", pytest.fail)
    off = _settings(db, inject_relevance=False)
    out = session_start(json.dumps({"source": "startup", "cwd": str(repo)}), off)
    assert _first_injected(out).startswith("Size GPU")


def test_session_start_still_injects_when_the_embedder_is_down(tmp_path):
    db = tmp_path / "index.db"
    _seed_tax_vs_gpu(db)
    repo = _git_repo(tmp_path / "ledger", {"CLAUDE.md": "tax lots and wash sales"})
    down = _settings(db, embed_backend="openai", embed_url="http://127.0.0.1:9/v1")
    out = session_start(json.dumps({"source": "startup", "cwd": str(repo)}), down)
    assert _first_injected(out).startswith("Size GPU")  # fell back to importance order
