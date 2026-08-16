"""Vault search: find a capture by what it *contains*, not what it is named.

Filenames tell you nothing about a photographed homework page, a screenshot,
or a scanned receipt — the only useful signal is the OCR text, the caption
FRIDAY generated for it, and the tags she assigned. This module searches that
signal, not the filesystem.

**Fusion, not a second ranker.** Two independent rankings of the same items
are combined here: a keyword pass over the index (always available) and,
when configured, a semantic vector pass. Rather than invent a bespoke scoring
scheme to merge them, this reuses :func:`friday.memory.rrf.reciprocal_rank_fusion`
— the same fusion primitive :class:`friday.memory.rrf.HybridRecall` uses for
long-term memory recall. RRF needs no score calibration between rankers (it
only consumes rank order), so a vault's naive substring keyword ranking and a
provider's cosine-similarity vector ranking combine correctly without either
side's raw scores being comparable. Every :class:`SearchHit` score returned
here is a genuine RRF score, not an invented positional one — the keyword-only
path (no vector store, or the store errored) still routes through RRF over
its single ranking, so scoring is always internally consistent.

**Degrade, don't fail.** The vector store is optional: a freshly created
vault, or one whose embedding backend is unavailable, has none. Search must
still work — keyword-only — because a vault the user cannot search at all is
worse than one that misses a few semantic matches. The same logic applies if
a *configured* vector store raises: it is an external dependency and its
outage must not take the whole search down with it.

**Locked items never appear, full stop.** This is the load-bearing guarantee.
Items are loaded via :meth:`~friday.vault.index.VaultIndex.list_items` with
``include_locked=False``, so the keyword ranking and the id universe used to
validate vector hits are both built from the owner's non-locked items only.
A vector store's hits are then filtered against that same non-locked id set
— so even if the store indexed a locked item's text independently (it is an
external system this module does not control), the id it returns for that
item can never reach a result, because it is not a member of the set being
searched. The same filter also prevents another owner's id — or any id that
simply is not the current owner's — from ever leaking through the vector
path.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from friday.logging import get_logger
from friday.memory.rrf import reciprocal_rank_fusion
from friday.vault.index import VaultIndex
from friday.vault.models import Item

logger = get_logger("friday.vault.search")

#: How many of the owner's items to load per search. A vault is personal
#: (photographed homework, receipts, screenshots), not shared or public-scale,
#: so a large fixed cap trades a bounded worst-case memory footprint against
#: never silently missing an old item because it fell outside a small
#: "most recent N" window — which is what passing a small ``limit`` straight
#: through to :meth:`~friday.vault.index.VaultIndex.list_items` would do. If a
#: vault ever meaningfully approaches this cap, that is a sign this scan-based
#: approach should be replaced with a real full-text index (e.g. SQLite FTS5)
#: rather than a sign to raise the constant further.
_MAX_ITEMS = 20_000

#: Snippet length is a display concern, not a matching one: enough to show
#: the user *why* an item matched without dumping the whole OCR blob.
_SNIPPET_LEN = 160


@runtime_checkable
class _VectorSearch(Protocol):
    """The shape this module needs from an optional vector store.

    Deliberately narrower than :class:`friday.memory.vector.VectorStore`
    (which returns scored ``Chunk`` objects keyed by ``source_id``) — vault
    search only needs a ranked list of item ids, so it accepts anything
    shaped like this rather than depending on that store's implementation.
    Route wiring for a concrete vector store lands in a later task.
    """

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


class SearchHit(BaseModel):
    """One matched item: id, fused rank score, and a preview of its text."""

    item_id: str
    score: float
    snippet: str


def _haystack_contains(item: Item, needle: str) -> bool:
    """Case-insensitive substring match over OCR text, caption, and tags.

    This is deliberately naive substring matching, not tokenized/word-level
    matching: a query is checked as one literal (lowercased) substring
    against each field. That means a multi-word query like "cell resistance"
    will not match text where both words are present but not adjacent in
    that order — word-level matching would fix that, but is out of scope
    here; the semantic vector pass is the intended mitigation for phrasing
    that substring matching misses.
    """
    if needle in item.ocr_text.lower():
        return True
    if needle in item.caption.lower():
        return True
    return any(needle in tag.lower() for tag in item.classification.tags)


def _snippet(text: str, length: int = _SNIPPET_LEN) -> str:
    text = text.strip()
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "…"


class VaultSearch:
    """Search one owner's vault by fusing keyword and (optional) vector recall.

    Args:
        index: The vault's metadata index.
        vector_store: An optional semantic ranker shaped like
            :class:`_VectorSearch`. ``None`` means keyword-only search.
    """

    def __init__(self, index: VaultIndex, *, vector_store: Any | None = None) -> None:
        self._index = index
        # Stored as `Any` (not `Any | None`): the `None` case is fully
        # handled by the `is not None` check in `search()` before
        # `_vector_rank` is ever called, and keeping the attribute itself
        # `Any` avoids mypy flagging a `.search()` call on a type that could
        # statically be `None`, without weakening the public `None`-accepting
        # constructor signature above.
        self._vector_store: Any = vector_store

    async def search(self, owner_uid: str, query: str, *, limit: int = 20) -> list[SearchHit]:
        """Return up to ``limit`` matches for ``query``, best first.

        Args:
            owner_uid: Whose vault to search — search never crosses owners.
            query: Free-text query. An empty or whitespace-only query yields
                no results (there is nothing to rank against).
            limit: Maximum number of hits to return. ``limit <= 0`` yields
                no results.

        Returns:
            Up to ``limit`` :class:`SearchHit` objects, best fused score
            first. Never includes a locked item, an item belonging to
            another owner, or an id not present in the vault.
        """
        query = query.strip()
        if not query or limit <= 0:
            return []

        items = self._index.list_items(owner_uid, include_locked=False, limit=_MAX_ITEMS)
        by_id = {item.id: item for item in items}
        needle = query.lower()

        keyword_ranking = [item.id for item in items if _haystack_contains(item, needle)]

        rankings = [keyword_ranking]
        if self._vector_store is not None:
            vector_ranking = self._vector_rank(query, limit, by_id)
            if vector_ranking:
                rankings.append(vector_ranking)

        fused = reciprocal_rank_fusion(rankings)

        hits: list[SearchHit] = []
        for item_id, score in fused[:limit]:
            item = by_id.get(item_id)
            if item is None:
                # Belt and suspenders: only the keyword ranking (built from
                # `items`) and the vector ranking (already filtered against
                # `by_id`) feed `rankings`, so this should be unreachable —
                # but a result is never emitted for an id we cannot resolve.
                continue
            hits.append(SearchHit(item_id=item_id, score=score, snippet=_snippet(item.ocr_text)))
        return hits

    def _vector_rank(self, query: str, limit: int, by_id: dict[str, Item]) -> list[str]:
        """Rank ids from the vector store, dropping anything not in ``by_id``.

        ``by_id`` is built from this owner's non-locked items only, so this
        is what stops a locked item's id, another owner's id, or a stale id
        the vector store still has indexed from ever reaching a result — the
        vector store's own output is not trusted on its own.

        A raising store degrades to an empty ranking (keyword-only overall)
        rather than failing the whole search: it is an external dependency
        and its outage should not take search down with it.
        """
        try:
            hits = self._vector_store.search(query, k=limit)
        except Exception:
            logger.warning("vault vector search failed; degrading to keyword-only", exc_info=True)
            return []
        return [item_id for item_id, _score in hits if item_id in by_id]
