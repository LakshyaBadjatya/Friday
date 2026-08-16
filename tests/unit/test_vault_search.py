"""Unit tests for vault search."""

from __future__ import annotations

import pytest

from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import CaptureSource, Classification, Item, Privacy
from friday.vault.search import VaultSearch


class _StubVectorStore:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._hits


class _RaisingVectorStore:
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise RuntimeError("vector backend unreachable")


def _index() -> SQLiteVaultIndex:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(
        Item(
            id="a",
            owner_uid="u1",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T10:00:00+00:00",
            ocr_text="a cell of emf 4 V and internal resistance 1 ohm",
            classification=Classification(subject="physics"),
        )
    )
    index.put_item(
        Item(
            id="b",
            owner_uid="u1",
            source=CaptureSource.SHARE,
            created_at="2026-08-16T11:00:00+00:00",
            ocr_text="grocery receipt total 840",
        )
    )
    return index


@pytest.mark.asyncio
async def test_keyword_hit_is_found_without_a_vector_store() -> None:
    results = await VaultSearch(_index(), vector_store=None).search("u1", "emf")
    assert [r.item_id for r in results] == ["a"]


@pytest.mark.asyncio
async def test_vector_and_keyword_hits_are_fused() -> None:
    search = VaultSearch(_index(), vector_store=_StubVectorStore([("b", 0.9)]))
    assert {r.item_id for r in await search.search("u1", "emf")} == {"a", "b"}


@pytest.mark.asyncio
async def test_locked_items_never_surface() -> None:
    index = _index()
    item = index.get_item("u1", "a")
    assert item is not None
    item.privacy = Privacy.LOCKED
    index.put_item(item)
    assert await VaultSearch(index, vector_store=None).search("u1", "emf") == []


@pytest.mark.asyncio
async def test_locked_item_id_from_vector_store_is_still_suppressed() -> None:
    """The most important guarantee: even if the vector store independently
    indexed a locked item's text and returns its id as a hit, that id must
    never reach the caller. The filter is "is this id in the owner's
    non-locked item set", not "trust the store" — this is what closes the
    hole.
    """
    index = _index()
    item = index.get_item("u1", "a")
    assert item is not None
    item.privacy = Privacy.LOCKED
    index.put_item(item)
    search = VaultSearch(index, vector_store=_StubVectorStore([("a", 0.99), ("b", 0.5)]))
    results = await search.search("u1", "emf")
    assert [r.item_id for r in results] == ["b"]


@pytest.mark.asyncio
async def test_another_owners_id_from_vector_store_never_leaks() -> None:
    """A vector store is not necessarily owner-scoped internally; the search
    layer must not trust ids it returns blindly. An id belonging to some
    other owner (or that simply does not exist in this owner's set) must be
    dropped, not surfaced.
    """
    index = _index()
    index.put_item(
        Item(
            id="stranger-item",
            owner_uid="someone-else",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T12:00:00+00:00",
            ocr_text="emf of a different battery entirely",
        )
    )
    search = VaultSearch(
        index, vector_store=_StubVectorStore([("stranger-item", 0.99), ("b", 0.5)])
    )
    results = await search.search("u1", "emf")
    assert [r.item_id for r in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_query_returns_no_results() -> None:
    assert await VaultSearch(_index(), vector_store=None).search("u1", "") == []


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_no_results() -> None:
    assert await VaultSearch(_index(), vector_store=None).search("u1", "   ") == []


@pytest.mark.asyncio
async def test_empty_vault_returns_no_results() -> None:
    index = SQLiteVaultIndex(":memory:")
    assert await VaultSearch(index, vector_store=None).search("u1", "emf") == []


@pytest.mark.asyncio
async def test_query_matching_nothing_returns_empty() -> None:
    results = await VaultSearch(_index(), vector_store=None).search("u1", "xylophone")
    assert results == []


@pytest.mark.asyncio
async def test_limit_zero_returns_no_results() -> None:
    results = await VaultSearch(_index(), vector_store=None).search("u1", "emf", limit=0)
    assert results == []


@pytest.mark.asyncio
async def test_case_insensitive_text_match() -> None:
    results = await VaultSearch(_index(), vector_store=None).search("u1", "EMF")
    assert [r.item_id for r in results] == ["a"]


@pytest.mark.asyncio
async def test_case_insensitive_tag_match() -> None:
    index = _index()
    index.put_item(
        Item(
            id="c",
            owner_uid="u1",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T12:00:00+00:00",
            classification=Classification(tags=["Circuit"]),
        )
    )
    results = await VaultSearch(index, vector_store=None).search("u1", "circuit")
    assert [r.item_id for r in results] == ["c"]


@pytest.mark.asyncio
async def test_query_with_regex_special_characters_is_not_treated_as_regex() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(
        Item(
            id="d",
            owner_uid="u1",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T12:00:00+00:00",
            ocr_text="resistor value (10 ohm) [tolerance 5%] a.b*c+",
        )
    )
    # An unbalanced/invalid pattern would raise re.error if compiled as regex.
    results = await VaultSearch(index, vector_store=None).search("u1", "(10 ohm")
    assert [r.item_id for r in results] == ["d"]
    results = await VaultSearch(index, vector_store=None).search("u1", "a.b*c+")
    assert [r.item_id for r in results] == ["d"]


@pytest.mark.asyncio
async def test_vector_store_error_degrades_to_keyword_only() -> None:
    """The vector store is an external dependency; a failure there must not
    take down search entirely — it should degrade to the keyword ranking.
    """
    search = VaultSearch(_index(), vector_store=_RaisingVectorStore())
    results = await search.search("u1", "emf")
    assert [r.item_id for r in results] == ["a"]


@pytest.mark.asyncio
async def test_multi_word_query_is_naive_substring_matching() -> None:
    """Documents a known limitation: the keyword pass matches the query as one
    literal substring, so a two-word query whose words both appear in the
    text (but not adjacently, in that order) is not found by keyword alone.
    Semantic (vector) search is the mitigation when a store is configured;
    this test pins the current keyword-only behaviour so a future change to
    it is deliberate, not accidental.
    """
    results = await VaultSearch(_index(), vector_store=None).search("u1", "resistance cell")
    assert results == []


@pytest.mark.asyncio
async def test_snippet_is_present_and_truncated() -> None:
    index = SQLiteVaultIndex(":memory:")
    long_text = "emf " + ("x" * 500)
    index.put_item(
        Item(
            id="e",
            owner_uid="u1",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T12:00:00+00:00",
            ocr_text=long_text,
        )
    )
    results = await VaultSearch(index, vector_store=None).search("u1", "emf")
    assert len(results) == 1
    assert len(results[0].snippet) < len(long_text)


@pytest.mark.asyncio
async def test_results_are_capped_at_limit() -> None:
    index = SQLiteVaultIndex(":memory:")
    for i in range(5):
        index.put_item(
            Item(
                id=f"item-{i}",
                owner_uid="u1",
                source=CaptureSource.CAMERA,
                created_at=f"2026-08-16T{10 + i:02d}:00:00+00:00",
                ocr_text="emf reading",
            )
        )
    results = await VaultSearch(index, vector_store=None).search("u1", "emf", limit=2)
    assert len(results) == 2
