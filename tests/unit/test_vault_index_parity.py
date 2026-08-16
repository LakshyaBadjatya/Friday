"""Parity tests: SQLiteVaultIndex and FirestoreVaultIndex must behave identically.

The VaultIndex Protocol exists precisely so that callers — including a config
flag that picks the backend at runtime — cannot tell which implementation
they are talking to. Any behavioural divergence between the two is therefore
a production bug waiting to happen, not a mere test gap, which is why this
suite runs the *same* assertions against both rather than trusting that two
separately maintained test files stay in lockstep as the models evolve.

Where the two backends legitimately differ (delete_item's return value — see
that test below), the divergence is asserted explicitly here so it's a
documented decision rather than something a reader has to rediscover.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from friday.vault.firestore_index import FirestoreVaultIndex
from friday.vault.index import SQLiteVaultIndex, VaultIndex
from friday.vault.models import (
    CaptureSource,
    Classification,
    CloudinaryAsset,
    Item,
    Note,
    Privacy,
)


class _FakeFirestoreTransport:
    """An in-memory stand-in for Firestore's REST surface.

    Persists PATCHed documents keyed by path, serves single-document and
    collection GETs, and honours DELETE — enough to let the parity suite
    exercise genuine put/get/list/delete round trips instead of hand-
    scripting every response the way the unit tests do.

    Firestore paths alternate collection/document segments (``vaults/u1``,
    ``items``, ``i1`` — collection, document-id-within-owner-doc... in our
    schema concretely ``vaults/{owner}/items/{id}``), so a path with an even
    number of ``/``-separated segments names a document and an odd number
    names a collection; that's how a GET is routed here. Pagination is
    deliberately not modelled — it has its own focused coverage in
    test_vault_firestore_index.py, and modelling it here would only add
    noise to tests whose purpose is cross-backend behavioural parity.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        if method == "PATCH":
            assert body is not None
            self._docs[path] = body
            return body
        if method == "DELETE":
            self._docs.pop(path, None)
            return {}
        if method == "GET":
            return self._get(path)
        raise ValueError(f"unsupported method {method}")

    def _get(self, path: str) -> Any:
        segments = path.split("/")
        if len(segments) % 2 == 0:
            return self._docs.get(path)
        prefix = path + "/"
        matches = sorted(
            key for key in self._docs if key.startswith(prefix) and "/" not in key[len(prefix) :]
        )
        documents = [self._docs[key] for key in matches]
        return {"documents": documents} if documents else {}


def _sqlite() -> VaultIndex:
    return SQLiteVaultIndex(":memory:")


def _firestore() -> VaultIndex:
    return FirestoreVaultIndex(_FakeFirestoreTransport())


_BACKENDS = pytest.mark.parametrize(
    "make_index", [_sqlite, _firestore], ids=["sqlite", "firestore"]
)


def _item(item_id: str = "i1", **kw: object) -> Item:
    base: dict[str, object] = {
        "id": item_id,
        "owner_uid": "u1",
        "source": CaptureSource.CAMERA,
        "created_at": "2026-08-16T10:00:00+00:00",
    }
    base.update(kw)
    return Item.model_validate(base)


def _note(note_id: str = "n1", **kw: object) -> Note:
    base: dict[str, object] = {
        "id": note_id,
        "owner_uid": "u1",
        "created_at": "2026-08-16T10:00:00+00:00",
    }
    base.update(kw)
    return Note.model_validate(base)


@_BACKENDS
def test_put_then_get_round_trips(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item())
    got = index.get_item("u1", "i1")
    assert got is not None
    assert got.owner_uid == "u1"
    assert got.source is CaptureSource.CAMERA


@_BACKENDS
def test_get_refuses_another_owner(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item())
    assert index.get_item("u2", "i1") is None


@_BACKENDS
def test_list_items_scoped_to_owner(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item("a", owner_uid="u1"))
    index.put_item(_item("b", owner_uid="u2"))
    assert [i.id for i in index.list_items("u1")] == ["a"]
    assert [i.id for i in index.list_items("u2")] == ["b"]


@_BACKENDS
def test_list_items_filters_by_subject(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item("a", classification=Classification(subject="physics")))
    index.put_item(_item("b", classification=Classification(subject="chemistry")))
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["a"]


@_BACKENDS
def test_list_items_filters_by_kind(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item("a", classification=Classification(kind="homework")))
    index.put_item(_item("b", classification=Classification(kind="notes")))
    assert [i.id for i in index.list_items("u1", kind="homework")] == ["a"]


@_BACKENDS
def test_list_items_filters_by_space(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item("a", space="private"))
    index.put_item(_item("b", space="shared"))
    assert [i.id for i in index.list_items("u1", space="shared")] == ["b"]


@_BACKENDS
def test_list_items_hides_locked_items_until_asked(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item("open"))
    index.put_item(_item("shut", privacy=Privacy.LOCKED))
    assert [i.id for i in index.list_items("u1")] == ["open"]
    assert {i.id for i in index.list_items("u1", include_locked=True)} == {"open", "shut"}


@_BACKENDS
def test_list_items_hides_locked_items_when_combined_with_subject_filter(
    make_index: Callable[[], VaultIndex],
) -> None:
    """The privacy guarantee: a locked item matching another filter stays hidden."""
    index = make_index()
    index.put_item(_item("open", classification=Classification(subject="physics")))
    index.put_item(
        _item(
            "shut",
            classification=Classification(subject="physics"),
            privacy=Privacy.LOCKED,
        )
    )
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["open"]
    assert {
        i.id for i in index.list_items("u1", subject="physics", include_locked=True)
    } == {"open", "shut"}


@_BACKENDS
def test_list_items_orders_newest_first_and_respects_limit(
    make_index: Callable[[], VaultIndex],
) -> None:
    index = make_index()
    index.put_item(_item("early", created_at="2026-08-16T09:00:00+00:00"))
    index.put_item(_item("late", created_at="2026-08-16T11:00:00+00:00"))
    index.put_item(_item("mid", created_at="2026-08-16T10:00:00+00:00"))
    assert [i.id for i in index.list_items("u1")] == ["late", "mid", "early"]
    assert [i.id for i in index.list_items("u1", limit=2)] == ["late", "mid"]


@_BACKENDS
def test_list_items_breaks_created_at_ties_by_id_descending(
    make_index: Callable[[], VaultIndex],
) -> None:
    index = make_index()
    same_ts = "2026-08-16T10:00:00+00:00"
    index.put_item(_item("a", created_at=same_ts))
    index.put_item(_item("c", created_at=same_ts))
    index.put_item(_item("b", created_at=same_ts))
    assert [i.id for i in index.list_items("u1")] == ["c", "b", "a"]


@_BACKENDS
def test_total_bytes_sums_committed_assets_including_locked(
    make_index: Callable[[], VaultIndex],
) -> None:
    index = make_index()
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    index.put_item(_item("open", cloudinary=asset))
    index.put_item(_item("shut", cloudinary=asset, privacy=Privacy.LOCKED))
    assert index.total_bytes("u1") == 2000


@_BACKENDS
def test_total_bytes_scoped_to_owner(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    index.put_item(_item("a", owner_uid="u1", cloudinary=asset))
    index.put_item(_item("b", owner_uid="u2", cloudinary=asset))
    assert index.total_bytes("u1") == 1000
    assert index.total_bytes("u2") == 1000


@_BACKENDS
def test_note_round_trips_and_is_owner_scoped(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_note(_note("n1", owner_uid="u1", markdown="hi"))
    index.put_note(_note("n2", owner_uid="u2", markdown="bye"))

    got = index.get_note("u1", "n1")
    assert got is not None
    assert got.markdown == "hi"
    assert index.get_note("u2", "n1") is None
    assert [n.id for n in index.list_notes("u1")] == ["n1"]
    assert [n.id for n in index.list_notes("u2")] == ["n2"]


@_BACKENDS
def test_put_is_an_upsert(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item())
    index.put_item(_item(ocr_text="v = ir"))
    got = index.get_item("u1", "i1")
    assert got is not None
    assert got.ocr_text == "v = ir"
    assert len(index.list_items("u1")) == 1


def test_delete_item_return_value_diverges_by_backend() -> None:
    """The one legitimate, deliberate difference between the two backends.

    SQLite reports whether a row actually existed (`rowcount > 0`).
    Firestore's DELETE returns success whether or not the document existed,
    and detecting the difference would cost an extra read on every delete —
    not worth it since no caller in this codebase branches on the return
    value today. Recorded here explicitly so that if this divergence ever
    disappears, or a new unintentional one appears elsewhere, it's obvious
    from this one test rather than discovered in production.
    """
    sqlite_index = _sqlite()
    firestore_index = _firestore()
    assert sqlite_index.delete_item("u1", "missing") is False
    assert firestore_index.delete_item("u1", "missing") is True


@_BACKENDS
def test_delete_item_actually_removes_it(make_index: Callable[[], VaultIndex]) -> None:
    index = make_index()
    index.put_item(_item())
    index.delete_item("u1", "i1")
    assert index.get_item("u1", "i1") is None
