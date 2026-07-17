"""code_roots config + backfill scoping (skip subagent/workflow sub-transcripts)."""

from __future__ import annotations

import json
from pathlib import Path

from qmx.config import Settings
from qmx.index import backfill_chats
from qmx.store import Store
from tests.fakes import FakeEmbedder

TRANSCRIPT = "\n".join(
    [
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": "real conversation turn"}}
        ),
        json.dumps(
            {"type": "assistant", "message": {"role": "assistant", "content": "an answer here"}}
        ),
    ]
)


def test_backfill_skips_subagent_and_workflow_transcripts(tmp_path):
    embedder = FakeEmbedder(dim=64)
    store = Store.open(tmp_path / "index.db", embedder.dim, "fake")
    proj = tmp_path / "projects" / "proj"
    proj.mkdir(parents=True)
    (proj / "main.jsonl").write_text(TRANSCRIPT)  # main session transcript -> indexed
    sub = proj / "main" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-x.jsonl").write_text(TRANSCRIPT)  # subagent -> skipped
    wf = proj / "main" / "workflows" / "wf_1"
    wf.mkdir(parents=True)
    (wf / "agent-y.jsonl").write_text(TRANSCRIPT)  # workflow -> skipped

    stats = backfill_chats(tmp_path / "projects", store, embedder)
    assert stats.files_scanned == 1
    assert stats.files_indexed == 1  # only main.jsonl, not the sub-transcripts
    store.close()


def test_code_roots_config_default_and_override():
    d = Settings.load(config_path=Path("/does/not/exist"), env={})
    assert d.code_roots == ()
    s = Settings.load(
        config_path=Path("/does/not/exist"),
        env={"QMX_CODE_ROOTS": "~/GitHub/Cruise/xtorch, ~/GitHub/Me/the-dsvolk/qmx"},
    )
    assert s.code_roots == ("~/GitHub/Cruise/xtorch", "~/GitHub/Me/the-dsvolk/qmx")
    assert s.as_dict()["code_roots"] == list(s.code_roots)
