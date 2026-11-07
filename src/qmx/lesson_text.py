"""Lesson text: the main rule (first sentence) of a statement, and the one-sentence lesson template.

A lesson is ``statement`` + ``detail``. The template: the statement is ONE sentence, the reusable
rule, with no ``[type]`` tag; the why, the specifics and anything else go in ``detail``. Every v2
judge prompt and check treats a lesson's first sentence as its main rule, so a statement that packs
a second rule after it hides that rule from the comparison (eval 2026-11-06: 22% of live statements
had 2+ sentences, and the judge missed restatements of their second sentence).
:func:`one_sentence` enforces the template on every save and merge (:mod:`qmx.learnings`,
:mod:`qmx.consolidate`); :func:`main_rule` is the same sentence boundary, used by the judge.

No qmx imports, so both :mod:`qmx.learnings` and :mod:`qmx.consolidate` can use it.
"""

from __future__ import annotations

import re
import unicodedata

# A leading type tag copied into a statement ("[howto] Track ..."): never part of the rule.
_TYPE_TAG = re.compile(r"^\s*\[(?:howto|mistake|decision)\]\s*", re.I)
# Abbreviations whose period does not end a sentence ("... (Model Training) vs. Multiplier ...").
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|vs|etc|approx|incl|cf|Rev|Proc|No|Inc|Fig|Sec)\.", re.I)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`'\"(\[])")
_FOLD = str.maketrans(
    {
        **{chr(c): "-" for c in range(0x2010, 0x2016)},  # hyphen .. horizontal bar
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",  # curly single quotes
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }  # curly double quotes
)
# A first sentence shorter than this many words is not split off ("Step 1. Run ..." stays whole):
# a two-word "rule" would be worse than a long statement.
MIN_RULE_WORDS = 4


def _fold(s: str) -> str:
    """NFKC, dash variants to '-', curly quotes to straight ones."""
    return unicodedata.normalize("NFKC", s).translate(_FOLD)


def main_rule(statement: str | None) -> str:
    """A lesson's main rule: the first sentence of its statement, without a leading ``[type]``
    tag; common abbreviations ("e.g.", "vs.", "incl.") do not end the sentence."""
    s = _TYPE_TAG.sub("", _fold(statement or "")).strip()
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\0"), s)
    return _SENT_END.split(protected, maxsplit=1)[0].replace("\0", ".").strip()


def strip_type_tags(statement: str | None) -> str:
    """``statement`` without leading ``[type]`` tags (any number), whitespace-trimmed."""
    s = (statement or "").strip()
    while True:
        out = _TYPE_TAG.sub("", s, count=1).strip()
        if out == s:
            return s
        s = out


def split_statement(statement: str | None) -> tuple[str, str]:
    """``(first sentence, rest)`` of a statement, tags stripped, characters kept as written.

    Same boundary as :func:`main_rule`: the quote/dash folding and the abbreviation guard are
    character-for-character, so a boundary found on the folded copy indexes the original.
    """
    s = strip_type_tags(statement)
    probe = _ABBREV.sub(lambda m: m.group(0).replace(".", "\0"), s.translate(_FOLD))
    m = _SENT_END.search(probe)
    if m is None:
        return s, ""
    return s[: m.start()].strip(), s[m.end() :].strip()


def _squash(s: str) -> str:
    return " ".join(s.split()).lower()


def one_sentence(statement: str, detail: str | None) -> tuple[str, str | None]:
    """Apply the lesson template to a ``statement``/``detail`` pair about to be saved.

    The statement keeps only its first sentence (leading ``[type]`` tags dropped); any further
    sentences move to the front of ``detail``, unless ``detail`` already contains them (so applying
    it twice, or to a merge that already moved them, changes nothing). Nothing is lost: injection,
    recall and embeddings all read statement + detail. A first sentence under
    :data:`MIN_RULE_WORDS` words is left as it is.
    """
    head, rest = split_statement(statement)
    if not rest or len(head.split()) < MIN_RULE_WORDS:
        return strip_type_tags(statement), detail
    old = (detail or "").strip()
    if _squash(rest) in _squash(old):
        return head, detail
    return head, f"{rest} {old}" if old else rest
