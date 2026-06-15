"""Unit tests for ``friday.memory.vector`` — the in-memory vector store.

The Phase-2 vector backend is a deterministic, dependency-free token-overlap
adapter behind a ``VectorStore`` protocol (real Chroma/pgvector is Phase 4).
These tests pin: an empty store returns ``[]``; after ``add`` a ``query``
returns the closest chunk carrying its ``source_id``; and similarity is
deterministic (token overlap), so the nearest document wins.
"""

from __future__ import annotations

from friday.memory.vector import Chunk, InMemoryVectorStore, VectorStore


def test_empty_store_returns_empty_list() -> None:
    store: VectorStore = InMemoryVectorStore()
    assert store.query("anything at all") == []


def test_add_then_query_returns_nearest_chunk_with_source_id() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            ("vector databases store embeddings for retrieval", "doc-vec"),
            ("the weather today is sunny and warm", "doc-weather"),
        ]
    )
    results = store.query("how do vector databases store embeddings", k=1)
    assert len(results) == 1
    top = results[0]
    assert isinstance(top, Chunk)
    assert top.source_id == "doc-vec"
    assert top.score > 0.0
    assert "vector" in top.text


def test_query_k_limits_and_orders_by_score() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            ("alpha beta gamma delta", "a"),
            ("alpha beta epsilon zeta", "b"),
            ("totally unrelated words here", "c"),
        ]
    )
    results = store.query("alpha beta gamma", k=2)
    assert len(results) == 2
    # Highest overlap first; doc "a" shares all three query tokens.
    assert results[0].source_id == "a"
    # Scores are sorted descending.
    assert results[0].score >= results[1].score


def test_query_is_deterministic() -> None:
    store = InMemoryVectorStore()
    store.add([("repeatable deterministic result", "d")])
    first = store.query("deterministic result")
    second = store.query("deterministic result")
    assert first == second


def test_no_overlap_yields_no_results() -> None:
    store = InMemoryVectorStore()
    store.add([("apples oranges bananas", "fruit")])
    # Query shares zero tokens -> nothing scores above zero -> empty.
    assert store.query("quantum chromodynamics lagrangian") == []
