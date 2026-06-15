"""Vector retrieval behind a small ``VectorStore`` interface (Phase-2 adapter).

This module introduces the retrieval seam used by the knowledge agent. The
contract is a :class:`VectorStore` protocol with :meth:`~VectorStore.add` and
:meth:`~VectorStore.query`; the Phase-2 implementation is
:class:`InMemoryVectorStore`, a dependency-free, fully deterministic adapter
that scores documents by Jaccard token overlap with the query.

Phase 4 adds :class:`SQLiteVectorStore`: a persistent, embedding-backed store
that keeps the same structural :class:`VectorStore` contract but ranks by cosine
similarity over vectors produced by an injected
:class:`~friday.providers.embeddings.EmbeddingProvider` (a deterministic fake in
tests, NVIDIA NIM in production). Both stores gain a ``forget`` operation. The
real backend is stdlib :mod:`sqlite3` — local-first, zero-server; Chroma /
pgvector remain documented adapter swaps, never required for the gate.

The in-memory adapter is intentionally simple and deterministic so retrieval is
reproducible in tests with no network, model download, or wall-clock surprises:
the same ``(documents, query)`` always yields the same ranked :class:`Chunk`
list, and documents sharing no tokens with the query never surface.
"""

from __future__ import annotations

import asyncio
import math
import re
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from friday.providers.embeddings import EmbeddingProvider

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

    def forget(self, query_or_source_id: str) -> int:
        """Drop every document matching ``query_or_source_id``; return the count.

        A document matches when its ``source_id`` equals the argument *or* its
        text contains the argument as a (case-insensitive) substring — mirroring
        a "forget what you know about X" command. Returns the number removed (0
        when nothing matched), and the dropped documents never surface again.
        """
        needle = query_or_source_id.lower()
        kept: list[tuple[str, str, frozenset[str]]] = []
        removed = 0
        for text, source_id, tokens in self._docs:
            if source_id == query_or_source_id or needle in text.lower():
                removed += 1
            else:
                kept.append((text, source_id, tokens))
        self._docs = kept
        return removed


# --------------------------------------------------------------------------- #
# Async-under-sync bridge
# --------------------------------------------------------------------------- #
def _run_coro[T](coro: Coroutine[object, object, T]) -> T:
    """Run ``coro`` to completion from synchronous code, loop-running or not.

    The :class:`VectorStore` contract is synchronous, but the embedding provider
    is async. When no event loop is running we drive the coroutine with
    :func:`asyncio.run`. When a loop *is* already running on this thread (e.g. an
    async agent calling ``store.query`` without awaiting), we cannot reenter it,
    so we run the coroutine to completion on a short-lived worker thread that
    owns its own loop. Either way the call is blocking and returns a value.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop on this thread — safe to drive one directly.
        return asyncio.run(coro)

    # A loop is already running here; offload to a thread with its own loop.
    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


# --------------------------------------------------------------------------- #
# Persistent SQLite-backed store (embedding + cosine)
# --------------------------------------------------------------------------- #
def _pack(vector: list[float]) -> bytes:
    """Serialize a float vector to a little-endian ``float64`` blob."""
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack(blob: bytes) -> list[float]:
    """Deserialize a ``float64`` blob back into a list of floats."""
    count = len(blob) // 8
    return list(struct.unpack(f"<{count}d", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b, strict=False):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SQLiteVectorStore:
    """A persistent, embedding-backed vector store over stdlib :mod:`sqlite3`.

    Implements the synchronous :class:`VectorStore` contract: :meth:`add` embeds
    each ``(text, source_id)`` pair with the injected
    :class:`~friday.providers.embeddings.EmbeddingProvider` and persists the text,
    source id, and packed embedding blob; :meth:`query` embeds the query and
    ranks stored rows by cosine similarity in Python, returning the top ``k``
    :class:`Chunk` objects closest-first. :meth:`forget` deletes rows by exact
    ``source_id`` or text substring and returns the number removed.

    The embedder is async while this contract is sync, so embedding calls are
    driven through :func:`_run_coro`, which is safe whether or not the caller is
    already inside an event loop. The store is local-first and zero-server: pass
    ``":memory:"`` for an ephemeral store or a file path for a persistent one.
    Concurrent use across threads is not assumed; each method opens, uses, and
    closes its own connection so reopening the same file always sees prior writes.
    """

    def __init__(self, path: str, embedder: EmbeddingProvider, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._path = path
        self._embedder = embedder
        self._dim = dim
        # ":memory:" connections vanish when closed, so a shared in-memory store
        # must hold one connection open for its lifetime; file stores reconnect
        # per call so a freshly opened store sees previously committed rows.
        self._shared: sqlite3.Connection | None = None
        if path == ":memory:":
            self._shared = sqlite3.connect(":memory:")
        self._init_schema()

    # -- connection plumbing ------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        return sqlite3.connect(self._path)

    def _release(self, conn: sqlite3.Connection) -> None:
        # Only per-call file connections are closed; the shared in-memory one
        # must stay open to retain its data.
        if self._shared is None:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  text TEXT NOT NULL,"
                "  source_id TEXT NOT NULL,"
                "  embedding BLOB NOT NULL"
                ")"
            )
            conn.commit()
        finally:
            self._release(conn)

    # -- VectorStore contract ----------------------------------------------- #
    def add(self, docs: list[tuple[str, str]]) -> None:
        """Embed and persist each ``(text, source_id)`` pair."""
        if not docs:
            return
        texts = [text for text, _ in docs]
        vectors = _run_coro(self._embedder.embed(texts))
        rows = [
            (text, source_id, _pack(vector))
            for (text, source_id), vector in zip(docs, vectors, strict=True)
        ]
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO chunks (text, source_id, embedding) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            self._release(conn)

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """Return up to ``k`` chunks ranked by cosine similarity, closest first.

        An empty store (or ``k <= 0``) yields ``[]``. Chunks with non-positive
        similarity are dropped so unrelated material never surfaces.
        """
        if k <= 0:
            return []
        conn = self._connect()
        try:
            stored = conn.execute(
                "SELECT text, source_id, embedding FROM chunks"
            ).fetchall()
        finally:
            self._release(conn)
        if not stored:
            return []

        (query_vector,) = _run_coro(self._embedder.embed([text]))
        scored: list[Chunk] = []
        for row_text, source_id, blob in stored:
            score = _cosine(query_vector, _unpack(blob))
            if score > 0.0:
                scored.append(Chunk(text=row_text, source_id=source_id, score=score))
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:k]

    def forget(self, query_or_source_id: str) -> int:
        """Delete rows matching ``query_or_source_id``; return the count removed.

        A row matches when its ``source_id`` equals the argument *or* its text
        contains the argument as a case-insensitive substring. Returns the number
        of rows deleted (0 when nothing matched).
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_id = ? "
                "OR instr(lower(text), lower(?)) > 0",
                (query_or_source_id, query_or_source_id),
            )
            removed = cursor.rowcount
            conn.commit()
        finally:
            self._release(conn)
        return removed
