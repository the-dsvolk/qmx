"""Reranking seam.

Phase 3 ships **RRF-only** ranking — no rerank stage — because Ollama exposes no rerank endpoint
and Qwen3-Reranker is not in its library (see ``plan/qmx-ml-notes.md``, TD-1). This module defines
the :class:`Reranker` protocol and a :class:`NoOpReranker` default so a real reranker (LLM-as-judge,
a dedicated ``/rerank`` server, or in-process cross-encoder) can be slotted into
:func:`qmx.search.search` later without changing call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qmx.search import RankedHit


@runtime_checkable
class Reranker(Protocol):
    """Reorders fused candidates by relevance to the query."""

    def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        """Return ``hits`` reordered best-first (may also drop/trim)."""
        ...


class NoOpReranker:
    """Passthrough — keeps the RRF order. The Phase 3 default."""

    def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        return hits
