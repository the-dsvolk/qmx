"""Fixed-layout judge grammars (qmx.grammar): exactly json.dumps(indent=2), no free whitespace.

The EBNF is checked live against vLLM/xgrammar when the layout changes (see the module docstring);
here its regex twin pins the *language*: it must accept every json.dumps(indent=2) rendering of a
valid reply and reject the free whitespace that let a prematurely closed string loop until the cap.
"""

from __future__ import annotations

import json
import re

import pytest

from qmx.consolidate import (
    CONSOLIDATE_SCHEMA,
    DEDUP_SCHEMA,
    DEPRECATE_SCHEMA,
    EXTRACT_SCHEMA,
    PRESERVE_SCHEMA,
)
from qmx.grammar import UnsupportedSchema, json_grammar, json_layout_regex

PRODUCTION = {
    "extract": EXTRACT_SCHEMA,
    "decision": CONSOLIDATE_SCHEMA,
    "merge": PRESERVE_SCHEMA,
    "retire": DEPRECATE_SCHEMA,
    "dedup": DEDUP_SCHEMA,
}

TRICKY = 'He said "use uv" \\ then: tab\there\nnewline, ünïcode'

SAMPLES = [
    (CONSOLIDATE_SCHEMA, {"reasoning": TRICKY, "action": "new"}),
    (CONSOLIDATE_SCHEMA, {"reasoning": "r", "action": "update", "target_id": 1907,
                          "statement": "s", "detail": "d", "importance": 0.85, "reason": ""}),
    (CONSOLIDATE_SCHEMA, {"reasoning": "r", "action": "drop", "target_id": None}),
    (CONSOLIDATE_SCHEMA, {"reasoning": "r", "action": "deprecate", "importance": -1.5e-3}),
    (EXTRACT_SCHEMA, {"reasoning": "nothing durable", "learnings": []}),
    (EXTRACT_SCHEMA, {"reasoning": "r", "learnings": [
        {"type": "howto", "statement": "s1"},
        {"type": "mistake", "topic": "t", "statement": "s2", "detail": TRICKY, "importance": 1},
    ]}),
    (PRESERVE_SCHEMA, {"reasoning": "r", "old_core_claim": "c", "evidence_in_new": "e",
                       "unsupported_additions": [], "preserves": True, "lost": ""}),
    (PRESERVE_SCHEMA, {"reasoning": "r", "old_core_claim": "c", "evidence_in_new": "",
                       "unsupported_additions": ["a", TRICKY], "preserves": False, "lost": "x"}),
    (DEPRECATE_SCHEMA, {"reasoning": "r", "contradiction_quote": "", "contradicts": False}),
    (DEDUP_SCHEMA, {"reasoning": "r", "same_lesson": True, "adds_new_fact": False,
                    "merged_statement": "", "merged_detail": ""}),
]


@pytest.mark.parametrize(("schema", "obj"), SAMPLES)
def test_accepts_exactly_the_indent2_rendering(schema, obj):
    rx = re.compile(json_layout_regex(schema))
    for ensure_ascii in (True, False):
        assert rx.fullmatch(json.dumps(obj, indent=2, ensure_ascii=ensure_ascii))
    assert not rx.fullmatch(json.dumps(obj))  # compact layout is not admitted
    assert not rx.fullmatch(json.dumps(obj, indent=4))


@pytest.mark.parametrize("name", list(PRODUCTION))
def test_no_free_whitespace_after_a_value(name):
    """The loop: string closed early, then newlines/tabs forever. Nothing may follow a value but
    the fixed separator."""
    rx = re.compile(json_layout_regex(PRODUCTION[name]))
    obj = next(o for s, o in SAMPLES if s is PRODUCTION[name])
    text = json.dumps(obj, indent=2)
    first_value_end = text.index('",') + 1  # end of the reasoning string
    for junk in ("\n", "\n\n\n", "\t", " "):
        looped = text[:first_value_end] + junk + text[first_value_end:]
        assert not rx.fullmatch(looped), repr(junk)


@pytest.mark.parametrize("name", list(PRODUCTION))
def test_grammar_keeps_reasoning_first_and_has_no_whitespace_class(name):
    g = json_grammar(PRODUCTION[name])
    root = g.splitlines()[0]
    assert root.startswith('root ::= "{\\n  \\"reasoning\\": " string')
    assert "[ \\n\\r\\t]" not in g and "\\s" not in g  # no free-whitespace rule anywhere
    assert {line.split(" ::= ")[0] for line in g.strip().splitlines()} >= {"root", "string"}


def test_rejects_wrong_order_missing_required_and_bad_values():
    rx = re.compile(json_layout_regex(CONSOLIDATE_SCHEMA))
    ok = {"reasoning": "r", "action": "update", "target_id": 3}
    assert rx.fullmatch(json.dumps(ok, indent=2))
    assert not rx.fullmatch(json.dumps({"action": "update", "reasoning": "r"}, indent=2))
    assert not rx.fullmatch(json.dumps({"reasoning": "r"}, indent=2))  # action is required
    assert not rx.fullmatch(json.dumps({**ok, "action": "merge"}, indent=2))  # not in the enum
    assert not rx.fullmatch(json.dumps({**ok, "target_id": "3"}, indent=2))  # integer|null
    assert not rx.fullmatch(json.dumps({**ok, "extra": 1}, indent=2))  # no extra properties
    bad_string = json.dumps(ok, indent=2).replace('"r"', '"line\nbreak"')  # raw control char
    assert not rx.fullmatch(bad_string)


def test_keys_and_enum_literals_are_escaped_for_ebnf():
    g = json_grammar({"type": "object", "properties": {"say \"hi\"": {"enum": ['a"b', 1, None]}},
                      "required": ['say "hi"']})
    assert '"{\\n  \\"say \\\\\\"hi\\\\\\"\\": "' in g  # JSON-escaped key, then EBNF-escaped
    assert '"\\"a\\\\\\"b\\""' in g and '"1"' in g and '"null"' in g


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"a": {"type": "string"}}},  # first property optional
    {"type": "object", "properties": {"a": {"type": "string", "pattern": "x"}}, "required": ["a"]},
    {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}, "required": ["a"]},
    {"type": "object", "properties": {"a": {"type": "array"}}, "required": ["a"]},  # no items
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
     "additionalProperties": True},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"]},
])
def test_outside_the_subset_raises(schema):
    with pytest.raises(UnsupportedSchema):
        json_grammar(schema)


def test_max_items_bounds_the_array():
    """The v2 merge check caps its logged lists at 3 (PRESERVE_SCHEMA_V2); the fixed layout must
    enforce the same bound as vLLM's own structured output does."""
    from qmx.consolidate import PRESERVE_SCHEMA_V2

    schema = {"type": "object", "properties": {"reasoning": {"type": "string"},
              "xs": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
              "required": ["reasoning", "xs"]}
    rx = re.compile(json_layout_regex(schema))
    for n in (0, 1, 2):
        assert rx.fullmatch(json.dumps({"reasoning": "r", "xs": ["a"] * n}, indent=2)), n
    assert not rx.fullmatch(json.dumps({"reasoning": "r", "xs": ["a"] * 3}, indent=2))
    json_grammar(PRESERVE_SCHEMA_V2)  # supported, so the opt-in layout still works for v2
    with pytest.raises(UnsupportedSchema):
        json_grammar({**schema, "properties": {**schema["properties"],
                      "xs": {"type": "array", "items": {"type": "string"}, "maxItems": -1}}})
