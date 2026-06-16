# © Lakshya Badjatya — Author
"""Citations & provenance: tie a grounded answer back to the chunks it used.

When FRIDAY answers from retrieved memory (RAG / the Knowledge path), every claim
should be traceable to its source. This module turns the retrieved chunks into
numbered references and attaches a provenance block to an answer, so the owner can
see exactly which stored source backed the reply — the honesty spine made visible.

It is pure and deterministic: sources are de-duplicated by ``source_id`` in first-
seen order and numbered ``[1]``, ``[2]``, … ; it imports no LLM SDK, reads no
configuration, and performs no I/O. The model still writes the prose; this module
only makes the grounding explicit.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel


class Source(BaseModel):
    """One retrieved chunk available to ground an answer.

    Attributes:
        source_id: Stable id of the chunk's origin (ties back to the store).
        text: The chunk text (used for an optional short snippet in the block).
    """

    source_id: str
    text: str = ""


class Citation(BaseModel):
    """A numbered reference assigned to a unique source."""

    marker: str
    source_id: str


class CitationFormatter:
    """Numbers sources and renders a provenance block.

    Args:
        snippet_chars: How many characters of a chunk to show in the references
            block (0 shows only the id). Default 0 (ids only).
    """

    def __init__(self, *, snippet_chars: int = 0) -> None:
        if snippet_chars < 0:
            raise ValueError("snippet_chars must be non-negative")
        self._snippet_chars = snippet_chars

    def references(self, sources: Iterable[Source]) -> list[Citation]:
        """Assign ``[1]``, ``[2]``, … to sources, de-duplicated by id (first-seen)."""
        citations: list[Citation] = []
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                continue
            seen.add(source.source_id)
            citations.append(
                Citation(marker=f"[{len(citations) + 1}]", source_id=source.source_id)
            )
        return citations

    def format_block(self, sources: Iterable[Source]) -> str:
        """Render a ``Sources:`` block; ``""`` when there are no sources."""
        source_list = list(sources)
        citations = self.references(source_list)
        if not citations:
            return ""
        by_id = {s.source_id: s for s in source_list}
        lines = ["Sources:"]
        for citation in citations:
            line = f"{citation.marker} {citation.source_id}"
            if self._snippet_chars:
                snippet = by_id[citation.source_id].text.strip().replace("\n", " ")
                if snippet:
                    line += f" — {snippet[: self._snippet_chars]}"
            lines.append(line)
        return "\n".join(lines)

    def attach(self, answer: str, sources: Iterable[Source]) -> str:
        """Append the provenance block to ``answer`` (unchanged when no sources)."""
        block = self.format_block(sources)
        if not block:
            return answer
        return f"{answer.rstrip()}\n\n{block}"
