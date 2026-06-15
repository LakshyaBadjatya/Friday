"""Vector retrieval behind a small ``VectorStore`` interface (Phase-2 adapter).

This module introduces the retrieval seam used by the knowledge agent. The
contract is a :class:`VectorStore` protocol with :meth:`~VectorStore.add` and
:meth:`~VectorStore.query`; the Phase-2 implementation is
:class:`InMemoryVectorStore`, a dependency-free, fully deterministic adapter
that scores documents by Jaccard token overlap with the query.

A real embedding-backed store (Chroma / pgvector) is deferred to Phase 4. The
in-memory adapter is intentionally simple and deterministic so retrieval is
reproducible in tests with no network, model download, or wall-clock surprises:
the same ``(documents, query)`` always yields the same ranked :class:`Chunk`
list, and documents sharing no tokens with the query never surface.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

# Split on any run of non-alphanumeric characters and lowercase, so tokenization
# is case-insensitive and punctuation-insensitive but otherwise byte-stable.
_TOKEN_RE = re.compile(r"[^0-9a-z]+")


def _tokenize(text: str) -> frozenset[str]:
    """Return the lowercase alphanumeric token set of ``text`` (empties dropped)."""
    return frozenset(tok for tok in _TOKEN_RE.split(text.lower()) if tok)


def _similarity(query_tokens: frozenset[str], doc_tokens: frozenset[str]) -> float:
    """Jaccard overlap of two token sets in ``[0.0, 1.0]`` (0 if either empty)."""
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = len(query_tokens & doc_tokens)
    if intersection == 0:
        return 0.0
    union = len(query_tokens | doc_tokens)
    return intersection / union


class Chunk(BaseModel):
    """A retrieved document chunk with its provenance and relevance score.

    ``source_id`` is the caller-supplied identifier passed to
    :meth:`VectorStore.add`; it lets a grounded answer cite exactly which
    document a passage came from. ``score`` is the (higher-is-closer) similarity
    of the chunk to the query.
    """

    text: str
    source_id: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Structural contract for a retrieval store.

    Implementations index ``(text, source_id)`` pairs via :meth:`add` and return
    the ``k`` most relevant :class:`Chunk` objects for a query via :meth:`query`.
    """

    def add(self, docs: list[tuple[str, str]]) -> None:
        """Index ``docs`` as ``(text, source_id)`` pairs."""
        ...

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """Return up to ``k`` chunks most relevant to ``text``, closest first."""
        ...


class InMemoryVectorStore:
    """A deterministic, in-process vector store using token-overlap similarity.

    No embeddings, no external services: documents are scored against the query
    by Jaccard overlap of their lowercase alphanumeric token sets. Results are
    returned highest-score-first; ties break by insertion order (Python's sort is
    stable), keeping output fully deterministic. Documents with zero overlap are
    omitted entirely so retrieval never returns spurious, unrelated chunks.
    """

    def __init__(self) -> None:
        # Each entry: (text, source_id, precomputed token set).
        self._docs: list[tuple[str, str, frozenset[str]]] = []

    def add(self, docs: list[tuple[str, str]]) -> None:
        """Index each ``(text, source_id)`` pair, precomputing its token set."""
        for text, source_id in docs:
            self._docs.append((text, source_id, _tokenize(text)))

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """Return the up-to-``k`` highest-overlap chunks for ``text``.

        Chunks with a zero similarity score are excluded. With an empty store,
        or when nothing overlaps the query, the result is an empty list.
        """
        if k <= 0:
            return []
        query_tokens = _tokenize(text)
        scored: list[Chunk] = []
        for doc_text, source_id, doc_tokens in self._docs:
            score = _similarity(query_tokens, doc_tokens)
            if score > 0.0:
                scored.append(
                    Chunk(text=doc_text, source_id=source_id, score=score)
                )
        # Stable sort by descending score; insertion order breaks ties.
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:k]
