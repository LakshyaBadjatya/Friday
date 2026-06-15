"""Unit tests for the persistent ``SQLiteVectorStore`` (and ``forget`` on both).

The persistent store embeds ``(text, source_id)`` pairs with an injected
:class:`~friday.providers.embeddings.EmbeddingProvider` (the deterministic
:class:`FakeEmbeddings` here — offline, no key) and ranks documents by cosine
similarity over the stored vectors. It implements the same sync
:class:`~friday.memory.vector.VectorStore` contract as the in-memory adapter,
plus a ``forget`` operation that drops rows matching a source id (or whose text
contains a query substring) and returns the number removed.

Pinned behaviour: empty store -> ``[]``; add-then-query returns the nearest
chunk carrying its ``source_id``; ``forget`` removes it and a re-query no longer
surfaces it; data survives reopening the same file (true persistence); and
``forget`` is also available on :class:`InMemoryVectorStore`.
"""

from __future__ import annotations

import asyncio

from friday.memory.vector import (
    Chunk,
    InMemoryVectorStore,
    SQLiteVectorStore,
    VectorStore,
)
from friday.providers.embeddings import FakeEmbeddings

DIM = 64


def _store(path: str) -> SQLiteVectorStore:
    return SQLiteVectorStore(path=path, embedder=FakeEmbeddings(dim=DIM), dim=DIM)


def test_sqlite_store_satisfies_protocol() -> None:
    store: VectorStore = _store(":memory:")
    assert isinstance(store, VectorStore)


def test_empty_store_returns_empty_list() -> None:
    store = _store(":memory:")
    assert store.query("anything at all") == []


def test_add_then_query_returns_nearest_chunk_with_source_id() -> None:
    store = _store(":memory:")
    store.add(
        [
            ("vector databases store embeddings for retrieval", "doc-vec"),
            ("the weather today is sunny and warm", "doc-weather"),
        ]
    )
    results = store.query("vector databases store embeddings for retrieval", k=1)
    assert len(results) == 1
    top = results[0]
    assert isinstance(top, Chunk)
    assert top.source_id == "doc-vec"
    assert top.text == "vector databases store embeddings for retrieval"


def test_query_k_limits_number_of_results() -> None:
    store = _store(":memory:")
    store.add(
        [
            ("alpha document one", "a"),
            ("beta document two", "b"),
            ("gamma document three", "c"),
        ]
    )
    results = store.query("alpha document one", k=2)
    assert len(results) == 2


def test_query_orders_by_descending_score() -> None:
    store = _store(":memory:")
    store.add(
        [
            ("alpha document one", "a"),
            ("beta document two", "b"),
            ("gamma document three", "c"),
        ]
    )
    results = store.query("alpha document one", k=3)
    scores = [chunk.score for chunk in results]
    assert scores == sorted(scores, reverse=True)


def test_query_k_zero_returns_empty() -> None:
    store = _store(":memory:")
    store.add([("alpha document one", "a")])
    assert store.query("alpha", k=0) == []


def test_forget_by_source_id_removes_and_requery_excludes() -> None:
    store = _store(":memory:")
    store.add(
        [
            ("vector databases store embeddings", "doc-vec"),
            ("python is a programming language", "doc-py"),
        ]
    )
    removed = store.forget("doc-vec")
    assert removed == 1
    results = store.query("vector databases store embeddings", k=4)
    assert all(chunk.source_id != "doc-vec" for chunk in results)


def test_forget_by_text_substring_removes_matching_rows() -> None:
    store = _store(":memory:")
    store.add(
        [
            ("the secret password is hunter2", "doc-secret"),
            ("an unrelated fact about cats", "doc-cats"),
        ]
    )
    removed = store.forget("password")
    assert removed == 1
    results = store.query("the secret password is hunter2", k=4)
    assert all("password" not in chunk.text for chunk in results)


def test_forget_unknown_returns_zero() -> None:
    store = _store(":memory:")
    store.add([("alpha document one", "a")])
    assert store.forget("nonexistent-source-id-xyz") == 0


def test_data_persists_across_reopen(tmp_path: object) -> None:
    db_file = str(tmp_path / "vec.db")  # type: ignore[operator]
    writer = _store(db_file)
    writer.add([("persisted knowledge survives restart", "doc-persist")])

    reader = _store(db_file)
    results = reader.query("persisted knowledge survives restart", k=1)
    assert len(results) == 1
    assert results[0].source_id == "doc-persist"


def test_query_is_deterministic() -> None:
    store = _store(":memory:")
    store.add([("repeatable deterministic result", "d")])
    first = store.query("repeatable deterministic result")
    second = store.query("repeatable deterministic result")
    assert first == second


def test_query_works_inside_running_event_loop() -> None:
    # The sync VectorStore facade must bridge the async embedder even when a
    # caller (e.g. an async agent) already has a running event loop.
    store = _store(":memory:")

    async def _exercise() -> list[Chunk]:
        store.add([("grounded inside a loop", "doc-loop")])
        return store.query("grounded inside a loop", k=1)

    results = asyncio.run(_exercise())
    assert len(results) == 1
    assert results[0].source_id == "doc-loop"


# --- InMemoryVectorStore.forget --------------------------------------------- #
def test_in_memory_forget_by_source_id() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            ("vector databases store embeddings", "doc-vec"),
            ("python is a programming language", "doc-py"),
        ]
    )
    removed = store.forget("doc-vec")
    assert removed == 1
    results = store.query("vector databases store embeddings")
    assert all(chunk.source_id != "doc-vec" for chunk in results)


def test_in_memory_forget_by_text_substring() -> None:
    store = InMemoryVectorStore()
    store.add([("the secret password is hunter2", "doc-secret")])
    removed = store.forget("password")
    assert removed == 1
    assert store.query("the secret password is hunter2") == []


def test_in_memory_forget_unknown_returns_zero() -> None:
    store = InMemoryVectorStore()
    store.add([("alpha", "a")])
    assert store.forget("nope") == 0
