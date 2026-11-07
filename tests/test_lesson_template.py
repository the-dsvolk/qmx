"""The one-sentence lesson template (qmx.lesson_text.one_sentence), applied on every save/merge."""

from __future__ import annotations

import json

import pytest

from qmx.consolidate import build_judge, consolidate_session, main_rule
from qmx.index import index_transcript
from qmx.learnings import add_learning, revert_learning, update_learning
from qmx.lesson_text import one_sentence, split_statement, strip_type_tags
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder

SCOPE = "acme/widgets"


@pytest.fixture
def store():
    embedder = FakeEmbedder()
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


# -- the pure function -------------------------------------------------------------------------


def test_a_one_sentence_statement_is_left_alone():
    assert one_sentence("Rotate the build cache weekly", "it grows by 2 GB a day") == (
        "Rotate the build cache weekly",
        "it grows by 2 GB a day",
    )
    assert one_sentence("Rotate the build cache weekly.", None) == (
        "Rotate the build cache weekly.",
        None,
    )


def test_extra_sentences_lead_the_detail():
    s = "Rotate the build cache weekly. It grows by 2 GB a day. Disk alerts fire at 90%."
    assert one_sentence(s, "set by the infra team") == (
        "Rotate the build cache weekly.",
        "It grows by 2 GB a day. Disk alerts fire at 90%. set by the infra team",
    )
    assert one_sentence(s, None) == (
        "Rotate the build cache weekly.",
        "It grows by 2 GB a day. Disk alerts fire at 90%.",
    )
    assert one_sentence(s, "  ") == (
        "Rotate the build cache weekly.",
        "It grows by 2 GB a day. Disk alerts fire at 90%.",
    )


@pytest.mark.parametrize(
    "statement",
    [
        "[howto] Rotate the build cache weekly",
        "  [decision] [howto] Rotate the build cache weekly",
        "[HOWTO]Rotate the build cache weekly",
    ],
)
def test_leading_type_tags_are_dropped(statement):
    assert one_sentence(statement, None) == ("Rotate the build cache weekly", None)
    assert strip_type_tags(statement) == "Rotate the build cache weekly"


@pytest.mark.parametrize(
    "statement",
    [
        "Prefer small images, e.g. distroless ones, for every service",
        "Pick the queue vs. the cron job by latency needs",
        "Pin the runtime to v0.27.1 and the client to 1.5.2 in every lockfile",
        "Cache hits rose 3.5x after the warm-up step was added",
        "Name files like report.final.csv so the loader finds them",
    ],
)
def test_abbreviations_versions_and_decimals_do_not_split(statement):
    assert one_sentence(statement, None) == (statement, None)


def test_a_short_first_sentence_is_not_split_off():
    s = "Step 1. Run the migration before deploying the new schema."
    assert one_sentence(s, None) == (s, None)


def test_applying_it_twice_changes_nothing():
    once = one_sentence("Rotate the build cache weekly. It grows fast.", "old note")
    assert one_sentence(*once) == once


def test_rest_already_in_the_detail_is_not_duplicated():
    # e.g. a merge that already moved the second sentence into the detail
    assert one_sentence(
        "Rotate the build cache weekly. It grows fast.", "Background: it grows fast. Set by infra."
    ) == ("Rotate the build cache weekly.", "Background: it grows fast. Set by infra.")


def test_characters_are_kept_as_written_and_the_boundary_matches_main_rule():
    s = "Quote \u201cexact\u201d error text in tickets. Paraphrases hide the cause."
    head, rest = split_statement(s)
    assert head == "Quote \u201cexact\u201d error text in tickets."
    assert rest == "Paraphrases hide the cause."
    assert main_rule(s) == 'Quote "exact" error text in tickets.'  # folded for comparison only


# -- every save and merge path -------------------------------------------------------------------


def test_add_learning_applies_the_template(store):
    s, embedder = store
    lid = add_learning(
        s,
        embedder,
        type="howto",
        scope=SCOPE,
        statement="[howto] Rotate the build cache weekly. It grows by 2 GB a day.",
        detail="set by the infra team",
    )
    le = s.get_learning(lid)
    assert (le.statement, le.detail) == (
        "Rotate the build cache weekly.",
        "It grows by 2 GB a day. set by the infra team",
    )


def test_manual_update_applies_the_template_to_the_stored_detail(store):
    s, embedder = store
    lid = add_learning(
        s,
        embedder,
        type="howto",
        statement="Rotate the build cache weekly",
        detail="set by the infra team",
    )
    le = update_learning(
        s, embedder, lid, statement="Rotate the build cache daily. Weekly filled the disk."
    )
    assert (le.statement, le.detail) == (
        "Rotate the build cache daily.",
        "Weekly filled the disk. set by the infra team",
    )
    # A one-sentence statement leaves the detail alone.
    le = update_learning(s, embedder, lid, statement="Rotate the build cache every day")
    assert le.detail == "Weekly filled the disk. set by the infra team"
    # An explicitly set detail receives the extra sentences.
    le = update_learning(
        s, embedder, lid, statement="Rotate the cache daily. Disk is small.", detail="new note"
    )
    assert (le.statement, le.detail) == ("Rotate the cache daily.", "Disk is small. new note")


def test_revert_restores_a_legacy_row_exactly(store):
    s, embedder = store
    lid = add_learning(s, embedder, type="howto", statement="Rotate the build cache weekly")
    legacy = "[howto] Rotate the build cache weekly. It grows by 2 GB a day."
    s.update_learning(lid, statement=legacy, revision_reason="test: legacy row")
    update_learning(s, embedder, lid, statement="Rotate the build cache daily")
    assert revert_learning(s, embedder, lid).statement == legacy  # history is not rewritten


def _session(store, tmp_path):
    s, embedder = store
    path = tmp_path / "sess.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"type": r, "message": {"role": r, "content": t}})
            for r, t in (("user", "build cache"), ("assistant", "noted"))
        )
    )
    index_transcript(path, s, embedder)
    return s.document_id("chat", str(path.resolve()))


OLD = "Rotate the build cache weekly"
CAND = {
    "type": "howto",
    "statement": "Rotate the build cache weekly on every runner",
    "detail": "runners share no disk",
    "importance": 0.6,
}


def test_judge_update_applies_the_template(store, tmp_path):
    s, embedder = store
    doc = _session(store, tmp_path)
    target = add_learning(
        s, embedder, type="howto", scope=SCOPE, statement=OLD, detail="it grows by 2 GB a day"
    )
    update = {
        "reasoning": "same rule",
        "action": "update",
        "target_id": target,
        "statement": OLD + " on every runner. Runners share no disk.",
        "detail": "it grows by 2 GB a day",
    }
    chat = FakeChat(extractions=[[dict(CAND)]], decisions=[update])
    res = consolidate_session(s, embedder, chat, doc, scope=SCOPE, judge=build_judge("v2"))
    assert res.updated == 1
    le = s.get_learning(target)
    assert (le.statement, le.detail) == (
        OLD + " on every runner.",
        "Runners share no disk. it grows by 2 GB a day",
    )


def test_dedup_merge_applies_the_template(store, tmp_path):
    s, embedder = store
    doc = _session(store, tmp_path)
    target = add_learning(
        s, embedder, type="howto", scope=SCOPE, statement=OLD, detail="it grows by 2 GB a day"
    )
    merged = {
        "same_lesson": True,
        "adds_new_fact": True,
        "merged_statement": OLD + ". Runners share no disk.",
        "merged_detail": "",
    }
    chat = FakeChat(extractions=[[dict(CAND)]], dup_checks=[merged])
    res = consolidate_session(
        s, embedder, chat, doc, scope=SCOPE, judge=build_judge("v2"), dedup_min_similarity=0.01
    )
    assert res.dedup_merged == 1
    le = s.get_learning(target)
    assert (le.statement, le.detail) == (OLD + ".", "Runners share no disk. it grows by 2 GB a day")
