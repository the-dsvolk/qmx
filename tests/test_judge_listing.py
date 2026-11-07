"""``judge_listing``: how the decision call lists the existing lessons ("inline" | "labelled")."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from qmx import consolidate as C
from qmx.config import Settings
from qmx.consolidate import build_judge, judge_config
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder


@pytest.fixture
def store():
    embedder = FakeEmbedder()
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _lesson(store, statement, detail=None, *, legacy=None):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement=statement, detail=detail)
    if legacy is not None:  # a row saved before the one-sentence template
        s.update_learning(lid, statement=legacy, revision_reason="test: legacy row")
    return s.get_learning(lid)


def test_default_is_inline_everywhere():
    assert Settings().judge_listing == "inline"
    assert judge_config(Settings()).listing == "inline"
    assert build_judge().listing == "inline"


def test_setting_values_are_validated_and_normalized():
    assert build_judge("v2", listing=" Labelled ").listing == "labelled"
    assert judge_config(dataclasses.replace(Settings(), judge_listing="labelled")).listing == (
        "labelled"
    )
    with pytest.raises(ValueError, match="judge_listing"):
        build_judge(listing="table")


def test_inline_listing_is_unchanged(store):
    le = _lesson(store, "Rotate the build cache weekly", "it grows by 2 GB a day")
    day = str(le.created_at)[:10]
    assert C._describe_match(le, 0.61) == (
        f"- id={le.learning_id} [howto] (similarity 0.61; created {day}) "
        "Rotate the build cache weekly (it grows by 2 GB a day)"
    )


def test_labelled_listing_shows_the_main_rule_and_the_detail(store):
    le = _lesson(store, "Rotate the build cache weekly", "it grows by 2 GB a day")
    day = str(le.created_at)[:10]
    assert C._describe_match_labelled(le, 0.61) == (
        f"- id={le.learning_id} [howto] (similarity 0.61; created {day})\n"
        "  main rule: Rotate the build cache weekly\n"
        "  detail: it grows by 2 GB a day"
    )


def test_labelled_listing_of_a_legacy_row_with_a_tag_and_two_sentences(store):
    le = _lesson(
        store,
        "Rotate the build cache weekly",
        legacy="[howto] Rotate the build cache weekly. It grows by 2 GB a day.",
    )
    lines = C._describe_match_labelled(le).splitlines()
    assert lines[1:] == [
        "  main rule: Rotate the build cache weekly.",
        "  statement: Rotate the build cache weekly. It grows by 2 GB a day.",
    ]


def test_labelled_listing_marks_a_retired_lesson(store):
    le = dataclasses.replace(
        _lesson(store, "Rotate the build cache weekly"),
        deprecated_at="2026-01-01",
        deprecated_reason="cache removed",
    )
    assert C._describe_match_labelled(le).splitlines()[-1] == "  [RETIRED: cache removed]"


@pytest.mark.parametrize("listing", ["inline", "labelled"])
def test_decide_sends_the_selected_listing(store, listing):
    le = _lesson(store, "Rotate the build cache weekly", "it grows by 2 GB a day")
    chat = FakeChat(decisions=[{"action": "new"}])
    cand = {"type": "howto", "statement": "Rotate the build cache on every runner"}
    C._decide(chat, cand, [le], {le.learning_id: 0.7}, judge=build_judge("v2", listing=listing))
    [prompt] = chat.decision_prompts
    assert ("  main rule: Rotate the build cache weekly" in prompt) is (listing == "labelled")
    assert ("Rotate the build cache weekly (it grows by 2 GB a day)" in prompt) is (
        listing == "inline"
    )


def test_listing_from_env_and_toml(tmp_path):
    env = {"QMX_JUDGE_LISTING": "labelled"}
    assert Settings.load(config_path=Path("/does/not/exist"), env=env).judge_listing == "labelled"
    cfg = tmp_path / "config.toml"
    cfg.write_text('judge_listing = "labelled"\n')
    assert Settings.load(config_path=cfg, env={}).judge_listing == "labelled"
