"""Judge filing rules v2 (2026-11-05): main rule in code, headline backstop, off-rule block,
two-quote retirement check, per-model reasoning wrapper — and the guarantee that the default
("current") judge is byte-identical to main @ 98bfd69.

Evidence for the design: ~/knowledge/qwen-judge-eval/prompt-research/ (FINDINGS.md, PROPOSAL.md
section A). The headline vectors are real old/new statements from the eval run stores.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging

import pytest

from qmx import cli
from qmx import consolidate as C
from qmx.config import Settings
from qmx.consolidate import (
    FILING_RULES,
    PRESERVE_SCHEMA_V2,
    QUOTE_RULE_V2,
    _clean_text,
    _merge_preserves,
    _norm_v2,
    build_judge,
    consolidate_session,
    headline_kept,
    judge_config,
    judge_prompts,
    main_rule,
)
from qmx.grammar import json_grammar
from qmx.index import index_transcript
from qmx.learnings import add_learning
from qmx.store import Store
from tests.fakes import FakeChat, FakeEmbedder, _call_kind
from tests.judge_scenario import GOLDEN_SHA256, Recorder, run_scenario, scenario_digest

SCOPE = "the-dsvolk/qmx"
V2 = build_judge("v2")  # Qwen wrapper (reasoning field)
V2_SUPER = build_judge("v2", reasoning_field=False, short_max_tokens=16384)
V2_BLOCK = build_judge("v2", offrule_block=True)  # judge_offrule_block = true


# == byte-identical defaults ======================================================================

# sha256 of each prompt and json.dumps(schema) on main @ 98bfd69 (computed before this change).
MAIN_SHA256 = {
    "EXTRACT": ("ce9b49b952ae137a81e061fec6935218b629ac3146fdd95e2e7eba9eacdfd4c8",
                "666eab7cd553efca0d40a5c680e76ebf3d57196610949d003e4164c4eeed2229"),
    "CONSOLIDATE": ("dc19cba677e7669cc01fd8e999413b3c767f638bac00ba0657f3b94fbd4adb87",
                    "4462571dad3b9a3d800ace9d4e3df61080a56d06afab1882e9b9cccb3cfdee38"),
    "PRESERVE": ("bc3ddb0accaed2a7b967f8e56839c5943291692a65dae8fb858fecbd2767145c",
                 "70e467e0025c0c2eb24c83e9398ce5585974b63a13b18984fba2a8b01a220c94"),
    "DEPRECATE": ("311be16e468196ec7d5eee4d3467be72b90fd762c83c47fc4d410c7f4711bcba",
                  "02a2c89366c6b84ef7d3c64c8336275960d842c8339ffad302ccc36fece3195e"),
    "DEDUP": ("2316cbcb4a9d16f2b82fa3d64b8bb705234310e61dc164ba1fffdd0477f602bc",
              "fb93b2b7bde11a9e411ee3f42dc8ba49816f65802bcbb98b48ea48da27ff4354"),
}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.mark.parametrize("name", list(MAIN_SHA256))
def test_current_prompt_constants_are_mains(name):
    system, schema = MAIN_SHA256[name]
    assert _sha(getattr(C, f"{name}_SYSTEM")) == system
    assert _sha(json.dumps(getattr(C, f"{name}_SCHEMA"))) == schema


def test_default_settings_select_todays_judge_exactly():
    j = judge_config(Settings())
    p = j.prompts
    assert (j.filing_rules, j.reasoning_field, j.v2) == ("current", True, False)
    assert j.short_max_tokens == C.SHORT_REPLY_MAX_TOKENS == 3072
    pairs = [(p.extract_system, C.EXTRACT_SYSTEM), (p.extract_schema, C.EXTRACT_SCHEMA),
             (p.consolidate_system, C.CONSOLIDATE_SYSTEM),
             (p.consolidate_schema, C.CONSOLIDATE_SCHEMA),
             (p.preserve_system, C.PRESERVE_SYSTEM), (p.preserve_schema, C.PRESERVE_SCHEMA),
             (p.deprecate_system, C.DEPRECATE_SYSTEM), (p.deprecate_schema, C.DEPRECATE_SCHEMA),
             (p.dedup_system, C.DEDUP_SYSTEM), (p.dedup_schema, C.DEDUP_SCHEMA)]
    assert all(built is const for built, const in pairs)


@pytest.mark.parametrize("judge", ["unset", "default-settings"])
def test_default_run_sends_mains_requests_and_leaves_mains_store(tmp_path, judge):
    """All five calls, request by request (system, user, schema, cap) + the final store, against
    the digest recorded on main @ 98bfd69."""
    kw = {} if judge == "unset" else {"judge": judge_config(Settings())}
    assert scenario_digest(tmp_path, **kw) == GOLDEN_SHA256


def test_unset_judge_honours_monkeypatched_constants(tmp_path, monkeypatch):
    # The eval harness patches module constants and calls consolidate_session without a judge.
    monkeypatch.setattr(C, "SHORT_REPLY_MAX_TOKENS", 1234)
    monkeypatch.setattr(C, "CONSOLIDATE_SYSTEM", C.CONSOLIDATE_SYSTEM + " PATCHED")
    calls, _ = run_scenario(tmp_path)
    decisions = [c for c in calls if _call_kind(c[0]) == "decision"]
    assert decisions and all(c[0].endswith(" PATCHED") for c in decisions)
    assert {c[3] for c in calls[1:]} == {1234}


def test_merge_preserves_keeps_its_two_tuple():
    # merge_check_probe2.py unpacks `ok, why = C._merge_preserves(...)`.
    class Old:
        learning_id, statement, detail = 1, "Use uv for Python envs", None

    chat = FakeChat()
    ok, why = _merge_preserves(chat, Old(), "Use uv for Python envs, always", None,
                               {"statement": "uv is fast"})
    assert (ok, why) == (True, "")


# == main rule and normalization ==================================================================


@pytest.mark.parametrize(("statement", "rule"), [
    ("First rule here. Second sentence.", "First rule here."),
    ("[howto] Do X with Y. Then Z.", "Do X with Y."),
    ("  [Decision]   Pick A over B! Because C.", "Pick A over B!"),
    ("Is it cached? Yes, for 5m.", "Is it cached?"),
    ("Compare billing (Model Training) vs. Multiplier per SKU. Then sum.",
     "Compare billing (Model Training) vs. Multiplier per SKU."),
    ("Size CUDs using billing percentiles (e.g. p05) rather than targets. Detail: x.",
     "Size CUDs using billing percentiles (e.g. p05) rather than targets."),
    ("N2D = 65.00% (77.95% total effective incl. CUD) matches the EA. Additionally, y.",
     "N2D = 65.00% (77.95% total effective incl. CUD) matches the EA."),
    ("Use foo. then bar it", "Use foo. then bar it"),  # lowercase: not a sentence start
    ("Do X. `cmd` does Y.", "Do X."),
    ("Do X. (Optional) Y.", "Do X."),
    ("Don’t use “uv pip” — ever. More.", "Don't use \"uv pip\" - ever."),
    ("Single sentence without a period", "Single sentence without a period"),
    ("", ""),
    (None, ""),
])
def test_main_rule_is_the_first_sentence(statement, rule):
    assert main_rule(statement) == rule


def test_norm_v2_folds_tags_quotes_case_and_dashes():
    assert _norm_v2("[howto]  Use `uv`  for “Python” – envs") == "use uv for python - envs"
    assert _norm_v2('Run "uv sync"') == _norm_v2("run 'uv sync'") == _norm_v2("Run uv sync")


# == headline backstop ============================================================================

# (old statement, new statement) pairs from real runs (eval stores, learning_revisions).
HEADLINE_PASS = {
    "BQ #1505 extension": (
        "[howto] Use `authcli` to mint a Cappy token for authenticating curl requests to dev "
        "environments.",
        "Use `authcli` to mint a Cappy token for authenticating curl requests to dev environments "
        "to verify deployments."),
    "P-Q #2003 extension after ;": (
        "[mistake] GitHub's 'merged' status on stacked PRs does not guarantee commits are in the "
        "target branch.",
        "GitHub's 'merged' status on stacked PRs does not guarantee commits are in the target "
        "branch; merging in order only advances intermediate branches unless the base branch is "
        "deleted to trigger retargeting."),
    "P-Q #2031 extension after ,": (
        "Store durable git worktrees in `~/worktrees/<repo>/<branch-slug>`.",
        "Store durable git worktrees in `~/worktrees/<repo>/<branch-slug>`, branched from a "
        "freshly fetched `origin/master`."),
    "P-S3 #1833 clause extension": (
        "[decision] When a capital loss carryover must absorb a sale's gain, sell the "
        "highest-cost-basis lots first.",
        "When a capital loss carryover must absorb a sale's gain, sell the highest-cost-basis lots "
        "first so the carryover covers more cash and more of it survives for later years. RSU "
        "basis was already taxed as salary at vest, so only the post-vest gain is capital gain."),
    "CGD #1157 same first sentence + new sentences": (
        "N2D discount in the GM/Google EA (Q-293542) is verified: On-Demand N2D = 65.00% off list "
        "and N2D VMs (1 Year CUD) = 65.00% (77.95% total effective incl. CUD) — the DRAFT ~/kb "
        "copy matches the executed pricing in force.",
        "N2D discount in the GM/Google EA (Q-293542) is verified: On-Demand N2D = 65.00% off list "
        "and N2D VMs (1 Year CUD) = 65.00% (77.95% total effective incl. CUD) — the DRAFT ~/kb "
        "copy matches the executed pricing in force. Additionally, because the 65% contract "
        "discount exceeds the standard ~37% list-price CUD discount, the incremental savings "
        "from CUDs on contracted N2D instances are significantly lower."),
    "double quotes rewritten as single (live #2079 pattern)": (
        'Run "uv sync --frozen" after every pull. Otherwise the lockfile drifts.',
        "Run 'uv sync --frozen' after every pull, before running tests."),
    "[howto] tag dropped": (
        "[howto] Pin the embedding model tag in config so reindexing stays reproducible.",
        "Pin the embedding model tag in config so reindexing stays reproducible."),
    "corrected number, same unit": (
        "Commit memory as `--resources=memory=88448GB` for the E2 pool.",
        "Commit memory as `--resources=memory=90570GB` for the E2 pool."),
    "corrected percentage and price": (
        "VPN egress is discounted 65% and costs $1,200.50 a month.",
        "VPN egress is discounted 40% and costs $980 a month; the rest is list price."),
    "identical": ("Use uv for Python envs.", "Use uv for Python envs."),
    "old statement empty": ("", "Anything at all."),
}

HEADLINE_FAIL = {
    "X-4057-med #2005 (seeded trap)": (
        "Model old ISO share sales by giving each Schedule D lot a second AMT cost basis (FMV at "
        "exercise from Form 3921), then compute the AMT adjustment (Form 6251 line 2k) and AMT "
        "capital-gains rates automatically. Exercising ISOs and selling the shares in the same "
        "calendar year triggers no AMT; the spread (sale price minus strike) is taxed as ordinary "
        "income, not as a capital gain.",
        "Exercising ISOs and selling in the same calendar year avoids AMT; the spread is taxed as "
        "ordinary income."),
    "X-4057-med #1979": (
        "Before ANY Snowflake (SNOW) sell advice for the household (sale, exercise-and-sell, "
        "harvest, rebalance, gift, any account), check the trade date against the dated "
        "trading-window schedule in ~/kb/personal/snowflake-trading-windows.md and state whether "
        "the window is open.",
        "During a closed insider trading window, a cash ISO exercise is allowed, but "
        "exercise-and-sell is not; the sale must wait for an open trading window."),
    "X-4057-med #1954": (
        "Ensure discoverability of new assets by updating both a local knowledge base file and the "
        "global KB index immediately after creation.",
        "[howto] Save analysis reports to the Reports folder, add a README row, register in the KB "
        "reports file, add a pointer and changelog entry in taxes.md, then refresh qmx."),
    "P-S2 #141": (
        "[howto] Load static prompts from a file in the script's directory at import time rather "
        "than passing them as arguments. Using `Path(__file__).parent / 'prompt.md'` ensures the "
        "prompt is always relative to the code.",
        "[howto] The call post-processing script takes one folder path, reads prompt.md from its "
        "own directory via Path(__file__).parent, runs the agent with cwd set to the folder, and "
        "writes summary.md into that folder."),
    "P-S2 #1933": (
        "[howto] Track wash-sale freezes across all household accounts, not just the primary "
        "brokerage. The IRS treats all accounts in a household (spouse, IRAs, RSUs) as one entity "
        "for wash-sale rules.",
        "Add a wash-sale adjustment input to taxc for wash sales between accounts."),
    "P-S2 #1329": (
        "[howto] When specifying memory in gcloud compute commitments, use GB as a proxy for GiB "
        "because the API interprets the input as binary units but returns the stored value in "
        "MiB. Detail: The command `--resources=memory=88448GB` resulted in `90570752` MiB.",
        "[howto] In `gcloud compute commitments create`, the `memory` flag is specified in GiB, "
        "not GB."),
    "P-S2 #1854": (
        "[howto] Verify file integrity and safety before/during copying. Perform pre-checks for "
        "source presence and destination absence to prevent overwrites.",
        "[howto] When moving files to a network share, copy first, verify checksums, then delete "
        "the source."),
    "P-S2 #1838": (
        "[howto] ESPP cost basis requires adjustment for tax reporting, not just the purchase "
        "price. For qualifying ESPP shares, the tax-adjusted basis includes the discount.",
        "ESPP shares have two cost bases: the purchase price on statements and a higher "
        "tax-adjusted basis that includes the discount taxed as ordinary income."),
    "P-S #1911": (
        "[howto] Size resource-based commitments using billing percentiles (e.g., p05) rather "
        "than recommender targets when usage is volatile. Detail: Recommender targets often "
        "assume different utilization floors.",
        "Size new CUD commitments to the p05 (95%-of-hours) uncovered usage rather than Google's "
        "always-on target, so the purchase almost never sits idle."),
    "P-S #1937": (
        "Implement `PRAGMA user_version` in the SQLite schema to manage database migrations and "
        "ensure backward compatibility.",
        "[decision] Use one persistent SQLite file for nvx outside the repo, with nightly backup "
        "and schema versioning via SQLite's built-in user_version plus numbered migration SQL."),
    "CGF #241": (
        "[howto] Handle audio format conversion explicitly when feeding buffers to "
        "`SpeechAnalyzer`.",
        "Handle audio format conversion explicitly for `AVAudioEngine` input nodes; install taps "
        "at the hardware format and convert with `AVAudioConverter`."),
    "narrowed (a word dropped)": (
        "Track wash-sale freezes across all household accounts.",
        "Track wash-sale freezes across all accounts."),
    "match must end on a word boundary": ("Use uv for envs.", "Use uv for envsets in CI."),
    "main rule demoted to detail (live #2074 pattern)": (
        "Cap judge replies at 3072 tokens. Longer replies are runaways.",
        "Judge replies can run away; see detail for the cap."),
}


@pytest.mark.parametrize("case", list(HEADLINE_PASS))
def test_headline_backstop_passes_extensions_and_corrected_numbers(case):
    old, new = HEADLINE_PASS[case]
    assert headline_kept(old, new)


@pytest.mark.parametrize("case", list(HEADLINE_FAIL))
def test_headline_backstop_blocks_known_clobbers(case):
    old, new = HEADLINE_FAIL[case]
    assert not headline_kept(old, new)


# == prompt builder: per-model wrapper, routing, schemas ==========================================

ALL_JUDGES = [(rules, rf) for rules in FILING_RULES for rf in (True, False)]


def _systems(p) -> dict[str, str]:
    return {"extract": p.extract_system, "decision": p.consolidate_system,
            "merge": p.preserve_system, "retire": p.deprecate_system, "dedup": p.dedup_system}


def _schemas(p) -> dict[str, dict]:
    return {"extract": p.extract_schema, "decision": p.consolidate_schema,
            "merge": p.preserve_schema, "retire": p.deprecate_schema, "dedup": p.dedup_schema}


@pytest.mark.parametrize(("rules", "rf"), ALL_JUDGES)
def test_routing_substrings_survive_in_every_variant(rules, rf):
    for kind, system in _systems(judge_prompts(rules, rf)).items():
        assert _call_kind(system) == kind


@pytest.mark.parametrize(("rules", "rf"), ALL_JUDGES)
def test_reasoning_field_switch_covers_all_five_calls(rules, rf):
    p = judge_prompts(rules, rf)
    for kind, schema in _schemas(p).items():
        props = list(schema["properties"])
        if rf:
            assert props[0] == "reasoning" and "reasoning" in schema["required"], kind
        else:
            assert "reasoning" not in props and "reasoning" not in schema["required"], kind
        json_grammar(schema)  # first property still required: the fixed layout stays available
    for kind, system in _systems(p).items():
        assert ("`reasoning`" in system) is rf, kind


def test_v2_shares_one_rule_text_across_both_wrappers():
    qwen, sup = judge_prompts("v2", True), judge_prompts("v2", False)
    as_super = (qwen.consolidate_system
                .replace(C._V2_WORK[True], C._V2_WORK[False])
                .replace(C._V2_RETURN[True], C._V2_RETURN[False]))
    assert as_super == sup.consolidate_system
    for name in ("preserve_system", "deprecate_system", "dedup_system"):
        assert getattr(qwen, name).replace(C._V2_THINK[True], "") == getattr(sup, name)
    assert sup.consolidate_system.endswith(
        "no [type] tags. Return JSON {action, target_id, statement, detail, importance, reason}."
        + QUOTE_RULE_V2)
    assert "Work through these steps before you answer: (1) Copy" in sup.consolidate_system
    assert ("Before deciding, work through these steps in the `reasoning` field: (1) Copy"
            in qwen.consolidate_system)


@pytest.mark.parametrize("rf", [True, False])
def test_v2_extraction_text_is_unchanged_and_checks_use_the_new_quote_rule(rf):
    v2, cur = judge_prompts("v2", rf), judge_prompts("current", rf)
    assert (v2.extract_system, v2.extract_schema) == (cur.extract_system, cur.extract_schema)
    assert v2.extract_system.endswith(C._QUOTE_RULE)
    for name in ("consolidate_system", "preserve_system", "deprecate_system", "dedup_system"):
        assert getattr(v2, name).endswith(QUOTE_RULE_V2)


def test_current_without_reasoning_field_matches_what_the_eval_harness_ran():
    p = judge_prompts("current", False)
    assert p.consolidate_system.endswith("or why you chose the action." + C._QUOTE_RULE)
    assert "Return JSON {action, target_id, statement, detail, importance, reason}. `statement`" \
        in p.consolidate_system
    assert p.extract_system.endswith("return an empty list if nothing is durable." + C._QUOTE_RULE)
    assert p.preserve_system.endswith("not really about a different subject." + C._QUOTE_RULE)


@pytest.mark.parametrize("rf", [True, False])
def test_v2_decision_has_no_lean_towards_new(rf):
    """Validation: "choose `new` ... when you cannot tell" made Qwen3.8 file same-rule candidates
    (keychain -> #123, list-price -> #1157) as new lessons."""
    decide = build_judge("v2", reasoning_field=rf).prompts.consolidate_system
    assert "cannot tell" not in decide
    assert "choose `new` only when they differ" in decide


def test_v2_merge_check_lists_are_capped_at_three():
    """Logged-only lists; uncapped, they ran long-lesson merge checks into the token cap."""
    props = PRESERVE_SCHEMA_V2["properties"]
    for name in ("lost_facts", "off_rule_additions", "unsupported_additions"):
        assert props[name]["maxItems"] == 3
    assert "up to three" in build_judge("v2").prompts.preserve_system


def test_v2_schemas():
    assert "old_core_claim" not in PRESERVE_SCHEMA_V2["properties"]
    assert list(PRESERVE_SCHEMA_V2["properties"]) == [
        "reasoning", "evidence_in_new", "lost_facts", "off_rule_additions",
        "unsupported_additions", "preserves", "lost"]
    assert PRESERVE_SCHEMA_V2["required"] == list(PRESERVE_SCHEMA_V2["properties"])
    p = judge_prompts("v2", False)
    assert p.deprecate_schema["required"] == ["contradiction_quote", "false_in_old", "contradicts"]
    assert p.consolidate_schema["required"] == ["action"]


def test_build_judge_validates_and_normalizes():
    assert build_judge(" V2 ").filing_rules == "v2"
    assert judge_config(Settings(judge_filing_rules="v2", judge_reasoning_field=False,
                                 chat_short_max_tokens=16384)) == V2_SUPER
    with pytest.raises(ValueError, match="judge_filing_rules"):
        build_judge("v3")


# == hygiene ======================================================================================


@pytest.mark.parametrize(("raw", "clean"), [
    ("[howto] Use uv for envs", "Use uv for envs"),
    ("'Make nvx review weekly'", "Make nvx review weekly"),
    ("'[decision] Don't wrap values'", "Don't wrap values"),
    ("'a' and 'b'", "'a' and 'b'"),  # two quoted words, not one wrapped value
    ("'users' files are private'", "'users' files are private'"),  # ambiguous: left alone
    ("plain", "plain"),
    (None, None),
])
def test_clean_text(raw, clean):
    assert _clean_text(raw) == clean


# == pipeline under v2 ============================================================================

OLD = "Track wash-sale freezes across all household accounts"
OLD_DETAIL = "the IRS treats spouse, IRAs and RSUs as one entity"
CAND = {"type": "howto", "statement": "Store wash-sale lots and trades in separate tables",
        "detail": "lots are snapshots, trades are events", "importance": 0.6}
CLOBBER = {"reasoning": "merge", "action": "update",
           "statement": "Add a wash-sale adjustment input to taxc", "detail": "for cross-account"}
EXTEND = {"reasoning": "merge", "action": "update",
          "statement": OLD + ", including IRAs", "detail": OLD_DETAIL + "; IRAs count"}
PASS_CHECK = {"preserves": True, "evidence_in_new": OLD, "lost_facts": [],
              "off_rule_additions": [], "unsupported_additions": [], "lost": ""}


@pytest.fixture
def store():
    embedder = FakeEmbedder(dim=64)
    with Store.open(":memory:", embedder.dim, "fake") as s:
        yield s, embedder


def _run(store, tmp_path, decision=None, *, judge=V2, statement=OLD, detail=OLD_DETAIL,
         cand=None, legacy_statement=None, **chat_kwargs):
    s, embedder = store
    path = tmp_path / "sess.jsonl"
    path.write_text("\n".join(json.dumps({"type": r, "message": {"role": r, "content": t}})
                              for r, t in (("user", "wash sales"), ("assistant", "noted"))))
    index_transcript(path, s, embedder)
    doc = s.document_id("chat", str(path.resolve()))
    target = add_learning(s, embedder, type="howto", scope=SCOPE, statement=statement,
                          detail=detail)
    if legacy_statement is not None:  # a row saved before the one-sentence template
        s.update_learning(target, statement=legacy_statement, revision_reason="test: legacy row")
    decisions = [{**decision, "target_id": target}] if decision else []
    chat = FakeChat(extractions=[[dict(cand or CAND)]], decisions=decisions, **chat_kwargs)
    rec = Recorder(chat)
    res = consolidate_session(s, embedder, rec, doc, scope=SCOPE, judge=judge,
                              dedup_min_similarity=0.01)  # always ask the near-duplicate check
    return s, target, chat, rec, res


def _kinds(rec) -> list[str]:
    return [_call_kind(c[0]) for c in rec.calls]


def test_headline_refused_update_gets_a_near_duplicate_second_attempt(store, tmp_path):
    s, target, chat, rec, res = _run(store, tmp_path, CLOBBER, dup_checks=[{"same_lesson": False}])
    assert _kinds(rec) == ["extract", "decision", "dedup"]  # no LLM merge check: code refused it
    assert (res.headline_blocked, res.update_blocked, res.updated, res.created) == (1, 0, 0, 1)
    assert s.get_learning(target).statement == OLD and s.learning_revisions(target) == []
    [new] = [le for le in s.list_learnings() if le.learning_id != target]
    assert (new.statement, new.detail) == (CAND["statement"], CAND["detail"])  # own words
    assert f"EXISTING MAIN RULE:\n{OLD}\n\nCANDIDATE:" in chat.dup_prompts[0]


def test_second_attempt_merge_that_keeps_the_headline_is_applied(store, tmp_path):
    merged = {"same_lesson": True, "adds_new_fact": True, "merged_statement": OLD + "; lots too",
              "merged_detail": OLD_DETAIL + "; lots are snapshots"}
    s, target, _, rec, res = _run(store, tmp_path, CLOBBER, dup_checks=[merged],
                                  checks=[PASS_CHECK])
    assert _kinds(rec) == ["extract", "decision", "dedup", "merge"]
    assert (res.headline_blocked, res.dedup_merged, res.created) == (1, 1, 0)
    assert s.get_learning(target).statement == OLD + "; lots too"


def test_near_duplicate_merge_that_rewrites_the_headline_is_dropped(store, tmp_path):
    clobber = {"same_lesson": True, "adds_new_fact": True,
               "merged_statement": "Store lots and trades separately", "merged_detail": "x"}
    s, target, _, rec, res = _run(store, tmp_path, None, dup_checks=[clobber])
    assert _kinds(rec) == ["extract", "decision", "dedup"]
    assert (res.headline_blocked, res.dedup_dropped, res.created) == (1, 1, 0)
    assert s.get_learning(target).statement == OLD


def test_off_rule_update_is_applied_and_flagged_by_default(store, tmp_path, caplog):
    """Default (judge_offrule_block = false): the headline is kept, so the merge goes through; the
    flag is a warning and a counter, not a refusal (validation: the block refused good merges)."""
    off = {**PASS_CHECK, "off_rule_additions": ["store lots and trades in separate tables"]}
    bloat = {**EXTEND, "statement": OLD + "; store lots and trades in separate tables"}
    with caplog.at_level(logging.WARNING, logger="qmx.consolidate"):
        s, target, _, rec, res = _run(store, tmp_path, bloat, checks=[off])
    assert _kinds(rec) == ["extract", "decision", "merge"]
    assert (res.offrule_flagged, res.offrule_blocked, res.update_blocked, res.updated,
            res.created) == (1, 0, 0, 1, 0)
    assert s.get_learning(target).statement == bloat["statement"]
    assert any("adds another rule" in r.getMessage() and "applied" in r.getMessage()
               for r in caplog.records)


def test_off_rule_update_is_stored_as_new_without_a_dedup_question(store, tmp_path):
    off = {**PASS_CHECK, "off_rule_additions": ["store lots and trades in separate tables"]}
    bloat = {**EXTEND, "statement": OLD + "; store lots and trades in separate tables"}
    s, target, _, rec, res = _run(store, tmp_path, bloat, checks=[off], judge=V2_BLOCK)
    assert _kinds(rec) == ["extract", "decision", "merge"]
    assert (res.offrule_blocked, res.update_blocked, res.created, res.updated) == (1, 1, 1, 0)
    assert res.offrule_flagged == 0
    assert s.get_learning(target).statement == OLD
    [new] = [le for le in s.list_learnings() if le.learning_id != target]
    assert new.statement == CAND["statement"]


def test_off_rule_near_duplicate_merge_keeps_the_candidate_as_new(store, tmp_path):
    """R2 #1761: a topic-match merge through the dedup path. The candidate was the session's key
    lesson — it must be stored, never dropped."""
    merged = {"same_lesson": True, "adds_new_fact": True,
              "merged_statement": OLD + ". Store lots and trades in separate tables.",
              "merged_detail": OLD_DETAIL}
    off = {**PASS_CHECK, "off_rule_additions": ["Store lots and trades in separate tables."]}
    s, target, _, rec, res = _run(store, tmp_path, None, dup_checks=[merged], checks=[off],
                                  judge=V2_BLOCK)
    assert _kinds(rec) == ["extract", "decision", "dedup", "merge"]
    assert (res.offrule_blocked, res.created, res.dedup_dropped, res.dedup_merged) == (1, 1, 0, 0)
    assert s.get_learning(target).statement == OLD and s.learning_revisions(target) == []
    assert {le.statement for le in s.list_learnings()} == {OLD, CAND["statement"]}


def test_off_rule_near_duplicate_merge_is_applied_by_default(store, tmp_path):
    merged = {"same_lesson": True, "adds_new_fact": True,
              "merged_statement": OLD + ". Store lots and trades in separate tables.",
              "merged_detail": OLD_DETAIL}
    off = {**PASS_CHECK, "off_rule_additions": ["Store lots and trades in separate tables."]}
    s, target, _, rec, res = _run(store, tmp_path, None, dup_checks=[merged], checks=[off])
    assert _kinds(rec) == ["extract", "decision", "dedup", "merge"]
    assert (res.offrule_flagged, res.offrule_blocked, res.dedup_merged, res.created) == (
        1, 0, 1, 0)
    # Lesson template: the merged statement keeps one sentence; the appended one leads the detail.
    after = s.get_learning(target)
    assert after.statement == OLD + "."
    assert after.detail == "Store lots and trades in separate tables. " + OLD_DETAIL


def test_claim_refusal_is_unchanged_under_v2(store, tmp_path):
    lost = {**PASS_CHECK, "preserves": False, "lost": "household scope"}
    s, target, _, _, res = _run(store, tmp_path, EXTEND, checks=[lost])
    assert (res.update_blocked, res.created, res.headline_blocked, res.offrule_blocked) == (
        1, 1, 0, 0)
    assert s.get_learning(target).statement == OLD


def test_merge_check_gets_old_main_rule_and_the_v2_schema(store, tmp_path, caplog):
    two_sentences = OLD + ". Brokers only track one account each."
    ext = {**EXTEND, "statement": two_sentences + " IRAs count too."}
    check = {**PASS_CHECK, "lost_facts": ["one entity"], "unsupported_additions": ["IRAs"]}
    with caplog.at_level(logging.WARNING, logger="qmx.consolidate"):
        s, target, chat, rec, res = _run(store, tmp_path, ext, legacy_statement=two_sentences,
                                         checks=[check])
    assert res.updated == 1  # lost facts / additions are logged, not blocked
    after = s.get_learning(target)  # lesson template: one sentence, the rest leads the detail
    assert after.statement == OLD + "."
    assert after.detail == ("Brokers only track one account each. IRAs count too. "
                            + EXTEND["detail"])
    [prompt] = chat.check_prompts
    assert f"OLD MAIN RULE:\n{OLD}.\n\nNEW:\n" in prompt  # the first sentence only
    [merge] = [c for c in rec.calls if _call_kind(c[0]) == "merge"]
    assert merge[2] == json.dumps(PRESERVE_SCHEMA_V2)
    assert "drops facts of the old lesson: one entity" in caplog.text
    assert "adds facts found in neither lesson: IRAs" in caplog.text


def test_corrected_number_in_the_headline_passes_the_backstop(store, tmp_path):
    old = "Commit memory as 88448GB for the E2 pool"
    fix = {"reasoning": "corrects the number", "action": "update",
           "statement": "Commit memory as 90570GB for the E2 pool", "detail": "binary units"}
    check = {**PASS_CHECK, "evidence_in_new": "Commit memory as 90570GB for the E2 pool"}
    s, target, _, _, res = _run(store, tmp_path, fix, statement=old, detail=None, checks=[check])
    assert (res.updated, res.headline_blocked) == (1, 0)


# -- two-quote retirement check --------------------------------------------------------------------

RETIRE_CAND = {"type": "decision", "statement": "Wash-sale freezes no longer apply to IRAs",
               "detail": "the 2026 rule change exempts retirement accounts", "importance": 0.7}
SUPERSEDE = {"reasoning": "stale", "action": "supersede", "statement": RETIRE_CAND["statement"],
             "detail": RETIRE_CAND["detail"]}
DEPRECATE = {"reasoning": "wrong", "action": "deprecate", "reason": "IRAs exempt"}


@pytest.mark.parametrize(("check", "verified"), [
    ({"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": "across all household accounts"}, True),
    ({"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": ""}, False),  # the old-side quote is required
    ({"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": "spouse, IRAs and RSUs"}, False),  # in OLD's detail, not its main rule
    ({"contradicts": True, "contradiction_quote": "IRAs are exempt from everything",
      "false_in_old": "across all household accounts"}, False),  # not in the candidate
    ({"contradicts": False, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": "across all household accounts"}, False),
    ({"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": "household"}, False),  # #2030 -> #2034: a one-word quote fits any rule
    ({"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
      "false_in_old": "all household accounts"}, True),  # three words is enough
], ids=["both-quotes", "no-old-quote", "old-quote-in-detail", "bad-candidate-quote", "no",
        "one-word-old-quote", "three-word-old-quote"])
def test_supersede_needs_both_quotes(store, tmp_path, check, verified):
    s, target, chat, _, res = _run(store, tmp_path, SUPERSEDE, cand=RETIRE_CAND,
                                   retire_checks=[check], dup_checks=[{"same_lesson": False}])
    assert (res.superseded, res.supersede_blocked) == ((1, 0) if verified else (0, 1))
    assert s.get_learning(target).is_live is (not verified)
    [prompt] = chat.retire_prompts
    assert f"OLD:\n{OLD}\ndetail: {OLD_DETAIL}\n\nOLD MAIN RULE:\n{OLD}\n\nCANDIDATE:\n" in prompt


@pytest.mark.parametrize("verified", [True, False])
def test_deprecate_needs_both_quotes(store, tmp_path, verified):
    check = {"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
             "false_in_old": "across all household accounts" if verified else ""}
    s, target, _, _, res = _run(store, tmp_path, DEPRECATE, cand=RETIRE_CAND,
                                retire_checks=[check])
    assert (res.deprecated, res.deprecate_blocked) == ((1, 0) if verified else (0, 1))
    assert res.created == (0 if verified else 1)


def test_short_main_rule_can_be_quoted_whole(store, tmp_path):
    """A main rule shorter than the minimum can still be retired by quoting all of it."""
    check = {"contradicts": True, "contradiction_quote": "no longer apply to IRAs",
             "false_in_old": "Freeze IRAs"}
    s, target, _, _, res = _run(store, tmp_path, SUPERSEDE, cand=RETIRE_CAND,
                                statement="Freeze IRAs", retire_checks=[check])
    assert (res.superseded, res.supersede_blocked) == (1, 0)


def test_quotes_are_compared_after_folding(store, tmp_path):
    check = {"contradicts": True, "contradiction_quote": "No longer apply to “IRAs”",
             "false_in_old": "ACROSS all `household` accounts"}
    _, _, _, _, res = _run(store, tmp_path, SUPERSEDE, cand=RETIRE_CAND, retire_checks=[check])
    assert res.superseded == 1


# -- hygiene, caps, wrapper end to end -------------------------------------------------------------


def test_v2_cleans_judge_written_text(store, tmp_path):
    new = {"reasoning": "novel", "action": "new",
           "statement": "'[howto] Store wash-sale lots and trades in separate tables'",
           "detail": "[howto] lots are snapshots"}
    s, target, _, _, res = _run(store, tmp_path, new)
    [le] = [le for le in s.list_learnings() if le.learning_id != target]
    assert (le.statement, le.detail) == (
        "Store wash-sale lots and trades in separate tables", "lots are snapshots")


def test_current_rules_store_judge_text_verbatim(store, tmp_path):
    # v2's quote unwrapping does not apply under the current rules...
    new = {"reasoning": "novel", "action": "new", "statement": "'Keep it as written'"}
    s, target, _, _, _ = _run(store, tmp_path, new, judge=None)
    assert "'Keep it as written'" in {le.statement for le in s.list_learnings()}


def test_lesson_template_strips_a_leading_tag_under_every_rule_set(store, tmp_path):
    # ...but the one-sentence lesson template (qmx.lesson_text) applies to every save.
    new = {"reasoning": "novel", "action": "new", "statement": "[howto] Keep the tag out"}
    s, target, _, _, _ = _run(store, tmp_path, new, judge=None)
    assert "Keep the tag out" in {le.statement for le in s.list_learnings()}


def test_super_wrapper_end_to_end(store, tmp_path):
    s, target, _, rec, res = _run(store, tmp_path, EXTEND, judge=V2_SUPER, checks=[PASS_CHECK])
    assert res.updated == 1
    assert _kinds(rec) == ["extract", "decision", "merge"]
    assert [c[3] for c in rec.calls] == [None, 16384, 16384]  # chat_short_max_tokens
    for system, _, schema, _ in rec.calls:
        assert "`reasoning`" not in system and '"reasoning"' not in schema


# -- CLI plumbing ----------------------------------------------------------------------------------


def _cli(tmp_path, monkeypatch, settings_kw, decision):
    embedder = FakeEmbedder(dim=64)
    db = tmp_path / "index.db"
    with Store.open(db, embedder.dim, "fake") as s:
        target = add_learning(s, embedder, type="howto", scope=SCOPE, statement=OLD,
                              detail=OLD_DETAIL)
    session = tmp_path / "sess.jsonl"
    session.write_text("\n".join(json.dumps({"type": r, "message": {"role": r, "content": t}})
                                 for r, t in (("user", "wash sales"), ("assistant", "ok"))))
    settings = Settings(db_path=db, embed_dim=embedder.dim, embed_model="fake",
                        ollama_url="http://127.0.0.1:9", consolidate_min_turns=0,
                        consolidate_min_chars=0, dedup_min_similarity=0.01, **settings_kw)
    rec = Recorder(FakeChat(extractions=[[dict(CAND)]],
                            decisions=[{**decision, "target_id": target}],
                            dup_checks=[{"same_lesson": False}]))
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, *a, **k: settings))
    monkeypatch.setattr(cli, "make_embedder", lambda _s: contextlib.nullcontext(embedder))
    monkeypatch.setattr(cli, "make_chat", lambda _s: contextlib.nullcontext(rec))
    code = cli.main(["consolidate", "--session", str(session), "--scope", SCOPE])
    return code, rec


def test_cli_runs_the_judge_the_settings_select(tmp_path, monkeypatch, capsys):
    code, rec = _cli(tmp_path, monkeypatch, {"judge_filing_rules": "v2",
                                             "judge_reasoning_field": False,
                                             "chat_short_max_tokens": 16384}, CLOBBER)
    assert code == 0
    systems = _systems(judge_prompts("v2", False))
    assert [c[0] for c in rec.calls] == [systems["extract"], systems["decision"], systems["dedup"]]
    assert [c[3] for c in rec.calls] == [None, 16384, 16384]
    out = capsys.readouterr().out
    assert "1 new, 0 updated" in out
    assert "1 merge(s) refused by the headline backstop" in out


def test_cli_default_settings_send_todays_prompts(tmp_path, monkeypatch, capsys):
    code, rec = _cli(tmp_path, monkeypatch, {}, CLOBBER)
    assert code == 0
    assert [c[0] for c in rec.calls] == [C.EXTRACT_SYSTEM, C.CONSOLIDATE_SYSTEM, C.PRESERVE_SYSTEM]
    assert [c[3] for c in rec.calls] == [None, 3072, 3072]
    assert "headline backstop" not in capsys.readouterr().out


def test_cli_rejects_unknown_filing_rules(tmp_path, monkeypatch, capsys):
    code, _ = _cli(tmp_path, monkeypatch, {"judge_filing_rules": "v3"}, CLOBBER)
    assert code == 1
    assert "judge_filing_rules" in capsys.readouterr().err
