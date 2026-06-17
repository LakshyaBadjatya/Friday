"""Document chunking + ingestion into the existing Phase-4 retrieval stores.

This module owns the *write* side of personal RAG. It introduces no new
retrieval path: it reuses the injected :class:`~friday.memory.vector.VectorStore`
(semantic chunks) and :class:`~friday.memory.long_term.LongTermStore` (a single
listable/forgettable "ingested <source>" marker fact). The matching *read* side
is the unchanged :class:`~friday.agents.knowledge.KnowledgeAgent`, which already
queries that same vector store and cites each chunk's ``source_id``.

Three pieces:

* :func:`chunk_text` — a paragraph/size-aware splitter with overlap that never
  drops content. It first splits on blank-line paragraph boundaries, then packs
  paragraphs into ``size``-bounded windows, carrying ``overlap`` characters of
  the previous window into the next so a fact spanning a boundary is still
  retrievable from at least one chunk.
* :class:`DocumentIngestor` — async ``ingest`` (chunk -> vector add under
  ``f"{source_id}#{i}"`` + one long-term marker fact), ``read_text`` (decode
  ``.txt`` / ``.md`` directly; ``.pdf`` via a *lazy* ``pypdf`` import with a
  clear pip-pointing error if absent), and ``forget_source`` (drop from both
  stores).
* :class:`IngestResult` — the typed ``{source_id, chunks}`` outcome.

No LLM SDK and no network: chunking is pure string work and the only async hop
is the injected embedding provider inside the vector store's ``add``.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

# Marker text recorded as a single long-term fact per ingested document. Kept in
# one place so listing (GET /rag/sources) and forgetting stay in lock-step.
_INGESTED_PREFIX = "ingested "

# Split on a run of two-or-more newlines (blank lines), the natural paragraph
# boundary; surrounding whitespace on each paragraph is stripped by the caller.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@runtime_checkable
class _ForgettableVectorStore(Protocol):
    """The slice of the vector-store contract ingestion writes through.

    Structural so any adapter (the in-memory or SQLite store) satisfies it: it
    must index ``(text, source_id)`` pairs and support forgetting by source id.
    """

    def add(self, docs: list[tuple[str, str]]) -> None: ...

    def forget(self, query_or_source_id: str) -> int: ...


@runtime_checkable
class _MarkerLongTermStore(Protocol):
    """The slice of the long-term contract used for the listable marker fact."""

    def add_fact(
        self, text: str, source_id: str, sensitive: bool = ...
    ) -> object: ...

    def forget(self, query: str) -> int: ...


class IngestResult(BaseModel):
    """The outcome of ingesting one document: its source id and chunk count."""

    source_id: str
    chunks: int


def _pack_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Pack ``paragraphs`` into ``size``-bounded, ``overlap``-linked windows.

    Paragraphs are accumulated into a window until adding the next would exceed
    ``size``; the window is then emitted and a fresh window seeded with the
    ``overlap`` trailing characters of the emitted text (so context spanning a
    boundary survives). A single paragraph longer than ``size`` is hard-split on
    word boundaries by :func:`_split_oversized` so no content is dropped.
    """
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            # Flush the in-progress window, then hard-split the oversized para.
            if current:
                chunks.append(current)
                current = _tail(current, overlap)
            for piece in _split_oversized(para, size, overlap):
                chunks.append((current + piece).strip() if current else piece)
                current = _tail(piece, overlap)
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > size:
            chunks.append(current)
            seed = _tail(current, overlap)
            current = f"{seed}\n\n{para}".strip() if seed else para
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_oversized(text: str, size: int, overlap: int) -> list[str]:
    """Hard-split a single oversized paragraph into overlapping word windows.

    Splitting is on word boundaries so a chunk never bisects a word; consecutive
    windows share ``overlap`` characters' worth of trailing words. No word is
    dropped — every word appears in at least one window.
    """
    words = text.split()
    if not words:
        return []
    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_len + added > size:
            windows.append(" ".join(current))
            # Seed the next window with trailing words covering ~``overlap`` chars.
            current, current_len = _overlap_words(current, overlap)
        current.append(word)
        current_len += len(word) + (1 if current_len else 0)
    if current:
        windows.append(" ".join(current))
    return windows


def _overlap_words(words: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing words of ``words`` covering up to ``overlap`` chars."""
    if overlap <= 0:
        return [], 0
    tail: list[str] = []
    length = 0
    for word in reversed(words):
        added = len(word) + (1 if tail else 0)
        if length + added > overlap:
            break
        tail.insert(0, word)
        length += added
    return tail, length


def _tail(text: str, overlap: int) -> str:
    """Return the last ``overlap`` characters of ``text``, on a word boundary.

    Used to seed the next window with the end of the previous one. The cut is
    snapped forward to the next word start so the carried text begins cleanly.
    """
    if overlap <= 0 or len(text) <= overlap:
        return text if len(text) <= overlap else ""
    snippet = text[-overlap:]
    space = snippet.find(" ")
    if space != -1:
        snippet = snippet[space + 1 :]
    return snippet.strip()


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into paragraph/size-aware overlapping chunks, losing nothing.

    Behaviour:

    * Empty / whitespace-only input -> ``[]`` (nothing to index).
    * Input that fits in ``size`` -> a single chunk equal to the stripped input.
    * Otherwise: paragraphs (blank-line separated) are packed into windows no
      larger than ``size``, with ``overlap`` characters carried between
      consecutive windows so a fact straddling a boundary is still wholly present
      in at least one chunk. An oversized single paragraph is hard-split on word
      boundaries. The concatenation of the chunks covers every word of the input.

    ``overlap`` is clamped to ``< size`` so progress is always made.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if size < 1:
        raise ValueError("size must be >= 1")
    overlap = max(0, min(overlap, size - 1))
    if len(stripped) <= size:
        return [stripped]
    paragraphs = [
        para.strip() for para in _PARAGRAPH_RE.split(stripped) if para.strip()
    ]
    if not paragraphs:
        return [stripped]
    return _pack_paragraphs(paragraphs, size, overlap)


class DocumentIngestor:
    """Ingest a document into the shared vector + long-term stores.

    A document is chunked by :func:`chunk_text` and each chunk is written to the
    injected :class:`~friday.memory.vector.VectorStore` under the id
    ``f"{source_id}#{i}"`` so a grounded answer can cite the exact passage. One
    "ingested <source_id>" marker fact is recorded in the long-term store so the
    document is listed by ``GET /rag/sources`` and removed by ``forget_source``.

    The ingestor depends only on the structural store slices (add/forget on the
    vector store; add_fact/forget on the long-term store), so any adapter works
    and no retrieval logic is duplicated here.

    Args:
        vector_store: The shared vector store ingested chunks are written to —
            the *same* store the knowledge agent retrieves from.
        long_term: The durable store the single marker fact is recorded in.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Characters carried between consecutive chunks.
    """

    def __init__(
        self,
        vector_store: _ForgettableVectorStore,
        long_term: _MarkerLongTermStore,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self._vector = vector_store
        self._long_term = long_term
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(self, source_id: str, text: str) -> IngestResult:
        """Chunk ``text`` and index it under ``source_id``; record one marker fact.

        Each chunk is added to the vector store as ``(chunk, f"{source_id}#{i}")``
        and a single ``"ingested {source_id}"`` fact is written to the long-term
        store (so the source is listed/forgettable). Returns an
        :class:`IngestResult` with the number of chunks indexed. Ingesting empty
        text indexes nothing but still records the marker so the (empty) source
        is listed and forgettable.
        """
        chunks = chunk_text(
            text, size=self._chunk_size, overlap=self._chunk_overlap
        )
        if chunks:
            docs = [
                (chunk, f"{source_id}#{i}") for i, chunk in enumerate(chunks)
            ]
            self._vector.add(docs)
        self._long_term.add_fact(
            f"{_INGESTED_PREFIX}{source_id}", source_id=source_id
        )
        return IngestResult(source_id=source_id, chunks=len(chunks))

    @staticmethod
    def read_text(filename: str, data: bytes) -> str:
        """Decode an uploaded ``filename``/``data`` pair into plain text.

        ``.txt`` / ``.md`` (and any unknown extension) are decoded as UTF-8
        (replacing undecodable bytes so ingestion never hard-fails on a stray
        byte). ``.pdf`` is read via a *lazy* ``pypdf`` import — ``pypdf`` is not a
        project dependency, so when it is absent a clear ``RuntimeError`` points
        the operator at ``pip install pypdf`` rather than leaking an ImportError.
        """
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return _read_pdf(data)
        return data.decode("utf-8", errors="replace")

    def forget_source(self, source_id: str) -> int:
        """Forget ``source_id`` from both stores; return total rows removed.

        Each ingested chunk is stored under the exact id ``f"{source_id}#{i}"``,
        so the chunks are dropped by forgetting those ids in order until one
        removes nothing (the contiguous index space ends). The long-term marker
        fact is removed too. Returns the combined count removed (0 when the
        source was never ingested).

        Note: we deliberately do NOT forget the bare ``source_id``. No chunk is
        ever stored under it (chunks use ``source_id#i``), so
        :meth:`VectorStore.forget` could only match it via its substring-of-text
        fallback — collaterally deleting chunks of *unrelated* documents whose
        text merely contains the source id string. The exact ``source_id#i`` loop
        below removes this source's own chunks precisely.
        """
        removed = 0
        index = 0
        while True:
            dropped = self._vector.forget(f"{source_id}#{index}")
            if dropped == 0:
                break
            removed += dropped
            index += 1
        removed += self._long_term.forget(f"{_INGESTED_PREFIX}{source_id}")
        return removed


def _read_pdf(data: bytes) -> str:
    """Extract text from a PDF byte payload via a lazy ``pypdf`` import.

    ``pypdf`` is optional and *not* in the project lock. It is imported here, at
    call time, so the dependency is required only when a ``.pdf`` is actually
    ingested; if it is missing we raise a clear ``RuntimeError`` naming the exact
    ``pip install`` to run, never a raw ``ImportError``.
    """
    import io

    try:
        import pypdf  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Reading PDF documents requires the optional 'pypdf' package, which "
            "is not installed. Install it with `pip install pypdf` (it is "
            "intentionally not a FRIDAY dependency), or convert the document to "
            "a .txt/.md file before ingesting."
        ) from exc

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
