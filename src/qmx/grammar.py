"""Fixed-layout JSON grammars for the judge on an OpenAI-compatible server (vLLM).

Why this exists. With ``response_format: json_schema`` vLLM compiles a grammar that allows *any
amount* of whitespace between JSON tokens. When the judge writes an unescaped ``"`` inside a string
(``...the core "rule" is...``), the string ends early; the grammar then allows only ``,``, ``}`` or
whitespace, the model wants to keep writing words, and greedy decoding emits newlines/tabs until
the token cap. Measured 2026-11-04 on replayed real sessions (Qwen3.8 NVFP4 on vLLM v0.33 /
xgrammar 0.2.8): about 9% of judge calls looped; one merge check timed out five times and deferred
a whole ``qmx consolidate --all``. Temperature does not fix it (sampling: 4/15 loops vs 5/15
greedy). vLLM's server-wide ``disable_any_whitespace`` stops the loop but forces compact JSON the
model rarely saw, and corrupted fields (``statement: ',detail'``, ``lost: '}]'``).

The fix: send, per request, a grammar that admits exactly ``json.dumps(obj, indent=2)`` layout. The
model keeps its natural tokens (``": "``, newline + indent) but there is no free whitespace to loop
on: after a premature close it must write ``,`` and the next key. Same language as xgrammar's
``Grammar.from_json_schema(schema, any_whitespace=False, indent=2, separators=(", ", ": "))``;
generated here because xgrammar (it pulls in torch) is too heavy a dependency for qmx. On the same
57 replayed calls: 0 loops, 0 corrupted fields, verdict agreement at the run-to-run noise level.

Supported JSON-schema subset (what qmx's judge schemas use): objects with ordered ``properties``
whose first property is required (it is always ``reasoning``), ``required``, ``type`` (string /
number / integer / boolean / null, or a list of them), ``enum`` of JSON scalars, and ``array`` with
``items`` (and an optional ``maxItems``). Anything else raises :class:`UnsupportedSchema`; the
client then falls back to ``response_format``. Property order is preserved, so ``reasoning`` stays
first (see consolidate).
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass

_INDENT = 2
_KEYWORDS = {"type", "properties", "required", "items", "enum", "description", "title",
             "additionalProperties", "maxItems"}

# Shared terminal rules (xgrammar EBNF syntax). A JSON string: no raw control characters, escapes
# per RFC 8259. ``\x00`` rather than ``\0``: both parse in xgrammar, this one is unambiguous.
_BASE_RULES = (
    'string ::= "\\"" string_char* "\\""',
    'string_char ::= [^\\x00-\\x1f"\\\\] | "\\\\" escape',
    'escape ::= ["\\\\/bfnrt] | "u" hex hex hex hex',
    "hex ::= [0-9a-fA-F]",
    'integer ::= "-"? ("0" | [1-9] [0-9]*)',
    'number ::= integer ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
)
_STRING_RE = r'"(?:[^\x00-\x1f"\\]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
_INTEGER_RE = r"-?(?:0|[1-9][0-9]*)"
_NUMBER_RE = _INTEGER_RE + r"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"


class UnsupportedSchema(ValueError):
    """The schema uses a feature outside the fixed-layout subset; use ``response_format``."""


@dataclass(frozen=True, slots=True)
class _G:
    """One grammar fragment, as xgrammar EBNF and as an equivalent Python regex (for tests)."""

    ebnf: str
    regex: str


def _lit(text: str) -> _G:
    if any(ord(c) < 0x20 and c not in "\n\r\t" for c in text):
        raise UnsupportedSchema(f"control character in literal {text!r}")
    esc = (text.replace("\\", "\\\\").replace('"', '\\"')
           .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return _G(f'"{esc}"', re.escape(text))


def _seq(*parts: _G) -> _G:
    return _G(" ".join(p.ebnf for p in parts), "".join(p.regex for p in parts))


def _alt(*parts: _G) -> _G:
    if len(parts) == 1:
        return parts[0]
    return _G("(" + " | ".join(f"({p.ebnf})" for p in parts) + ")",
              "(?:" + "|".join(p.regex for p in parts) + ")")


def _opt(part: _G) -> _G:
    return _G(f"({part.ebnf})?", f"(?:{part.regex})?")


def _star(part: _G) -> _G:
    return _G(f"({part.ebnf})*", f"(?:{part.regex})*")


_STRING = _G("string", _STRING_RE)
_INTEGER = _G("integer", _INTEGER_RE)
_NUMBER = _G("number", _NUMBER_RE)
_BOOLEAN = _alt(_lit("true"), _lit("false"))
_NULL = _lit("null")


def _value(schema: dict, depth: int) -> _G:
    if not isinstance(schema, dict):
        raise UnsupportedSchema(f"schema must be an object, got {schema!r}")
    extra = set(schema) - _KEYWORDS
    if extra:
        raise UnsupportedSchema(f"unsupported keyword(s): {sorted(extra)}")
    if "enum" in schema:
        values = schema["enum"]
        if not values or not all(v is None or isinstance(v, (str, int, float, bool))
                                 for v in values):
            raise UnsupportedSchema(f"enum must list JSON scalars: {values!r}")
        return _alt(*(_lit(json.dumps(v)) for v in values))
    kind = schema.get("type")
    if isinstance(kind, list):
        return _alt(*(_value({**schema, "type": k}, depth) for k in kind))
    if kind == "string":
        return _STRING
    if kind == "integer":
        return _INTEGER
    if kind == "number":
        return _NUMBER
    if kind == "boolean":
        return _BOOLEAN
    if kind == "null":
        return _NULL
    if kind == "array":
        return _array(schema, depth)
    if kind == "object":
        return _object(schema, depth)
    raise UnsupportedSchema(f"unsupported type {kind!r}")


def _array(schema: dict, depth: int) -> _G:
    if "items" not in schema:
        raise UnsupportedSchema("array without items")
    inner = _value(schema["items"], depth + 1)
    pad_in = "\n" + " " * (_INDENT * (depth + 1))
    pad_out = "\n" + " " * (_INDENT * depth)
    more = _seq(_lit("," + pad_in), inner)
    limit = schema.get("maxItems")
    if limit is None:
        rest = _star(more)
    else:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise UnsupportedSchema(f"maxItems must be a non-negative integer: {limit!r}")
        if limit == 0:
            return _lit("[]")
        rest = _G("", "")  # up to limit-1 more items, as nested optionals
        for _ in range(limit - 1):
            rest = _opt(_seq(more, rest))
    full = _seq(_lit("[" + pad_in), inner, rest, _lit(pad_out + "]"))
    return _alt(full, _lit("[]"))


def _object(schema: dict, depth: int) -> _G:
    if schema.get("additionalProperties", False) is not False:
        raise UnsupportedSchema("additionalProperties must be absent or false")
    props = schema.get("properties") or {}
    if not props:
        return _lit("{}")
    required = set(schema.get("required") or [])
    unknown = required - set(props)
    if unknown:
        raise UnsupportedSchema(f"required names unknown properties: {sorted(unknown)}")
    names = list(props)
    if names[0] not in required:
        # json.dumps never writes a leading comma, so the first emitted property decides the
        # comma layout; qmx schemas always lead with a required `reasoning`.
        raise UnsupportedSchema(f"first property {names[0]!r} must be required")
    pad_in = "\n" + " " * (_INDENT * (depth + 1))
    pad_out = "\n" + " " * (_INDENT * depth)
    parts = [_lit("{" + pad_in + json.dumps(names[0]) + ": "), _value(props[names[0]], depth + 1)]
    for name in names[1:]:
        member = _seq(_lit("," + pad_in + json.dumps(name) + ": "), _value(props[name], depth + 1))
        parts.append(member if name in required else _opt(member))
    parts.append(_lit(pad_out + "}"))
    return _seq(*parts)


@functools.lru_cache(maxsize=32)
def _build(schema_json: str) -> _G:
    return _value(json.loads(schema_json), 0)


def json_grammar(schema: dict) -> str:
    """xgrammar EBNF admitting exactly ``json.dumps(obj, indent=2)`` for objects valid under
    ``schema``. Raises :class:`UnsupportedSchema` outside the supported subset."""
    root = _build(json.dumps(schema))
    return "\n".join((f"root ::= {root.ebnf}", *_BASE_RULES)) + "\n"


def json_layout_regex(schema: dict) -> str:
    """The same language as :func:`json_grammar`, as a Python regex (for tests/diagnostics)."""
    return _build(json.dumps(schema)).regex
