"""Unit tests for the Firestore vault index, against a stubbed transport."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from friday.vault.firestore_index import FirestoreIndexError, FirestoreVaultIndex
from friday.vault.models import (
    CaptureSource,
    Classification,
    CloudinaryAsset,
    Item,
    Note,
    Privacy,
    Solve,
)


class _StubTransport:
    """Records calls and replays scripted responses — no network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.responses: dict[str, Any] = {}

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append((method, path, body))
        return self.responses.get(path)


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


def _solve(solve_id: str = "s1", **kw: object) -> Solve:
    base: dict[str, object] = {
        "id": solve_id,
        "item_ids": ["i1"],
        "created_at": "2026-08-16T10:00:00+00:00",
    }
    base.update(kw)
    return Solve.model_validate(base)


def _doc(model: Item | Note | Solve) -> dict[str, Any]:
    """Wrap a model's JSON the way a real Firestore document response would."""
    return {"fields": {"doc": {"stringValue": model.model_dump_json()}}}


def _listing(*models: Item | Note | Solve) -> dict[str, Any]:
    return {"documents": [_doc(m) for m in models]}


# --------------------------------------------------------------------------
# Required minimum (from the task draft).
# --------------------------------------------------------------------------


def test_put_item_patches_the_owner_scoped_document() -> None:
    transport = _StubTransport()
    FirestoreVaultIndex(transport).put_item(_item())
    method, path, body = transport.calls[0]
    assert method == "PATCH"
    assert path == "vaults/u1/items/i1"
    assert body is not None
    assert body["fields"]["doc"]["stringValue"].startswith("{")


def test_get_item_decodes_the_stored_document() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = _doc(_item())
    got = FirestoreVaultIndex(transport).get_item("u1", "i1")
    assert got is not None
    assert got.id == "i1"


def test_get_item_returns_none_when_absent() -> None:
    assert FirestoreVaultIndex(_StubTransport()).get_item("u1", "missing") is None


def test_delete_item_issues_a_delete() -> None:
    transport = _StubTransport()
    FirestoreVaultIndex(transport).delete_item("u1", "i1")
    assert transport.calls[0][0] == "DELETE"
    assert transport.calls[0][1] == "vaults/u1/items/i1"


# --------------------------------------------------------------------------
# Extra coverage beyond the required minimum.
# --------------------------------------------------------------------------


def test_list_items_filters_by_subject() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("a", classification=Classification(subject="physics")),
        _item("b", classification=Classification(subject="chemistry")),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["a"]


def test_list_items_filters_by_kind() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("a", classification=Classification(kind="homework")),
        _item("b", classification=Classification(kind="notes")),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1", kind="homework")] == ["a"]


def test_list_items_filters_by_space() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("a", space="private"),
        _item("b", space="shared"),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1", space="shared")] == ["b"]


def test_list_items_hides_locked_items_until_asked() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("open"),
        _item("shut", privacy=Privacy.LOCKED),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1")] == ["open"]
    assert {i.id for i in index.list_items("u1", include_locked=True)} == {"open", "shut"}


def test_list_items_hides_locked_items_when_combined_with_subject_filter() -> None:
    """A locked item matching the subject filter must still be hidden.

    Guards against a filter pipeline that short-circuits on the first match
    and forgets to also check privacy.
    """
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("open", classification=Classification(subject="physics")),
        _item(
            "shut",
            classification=Classification(subject="physics"),
            privacy=Privacy.LOCKED,
        ),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["open"]
    assert {
        i.id for i in index.list_items("u1", subject="physics", include_locked=True)
    } == {"open", "shut"}


def test_list_items_orders_newest_first_and_respects_limit() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(
        _item("early", created_at="2026-08-16T09:00:00+00:00"),
        _item("late", created_at="2026-08-16T11:00:00+00:00"),
        _item("mid", created_at="2026-08-16T10:00:00+00:00"),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1")] == ["late", "mid", "early"]
    assert [i.id for i in index.list_items("u1", limit=2)] == ["late", "mid"]


def test_list_items_breaks_created_at_ties_by_id_descending() -> None:
    """Items sharing a timestamp must sort deterministically, matching SQLite.

    A collection listing comes back in unspecified order, so without a
    secondary key on id, the sort would be an unstable reflection of
    whatever order the (stubbed, here-fixed) response happened to list
    documents in — a real Firestore response has no such guarantee at all.
    """
    transport = _StubTransport()
    same_ts = "2026-08-16T10:00:00+00:00"
    transport.responses["vaults/u1/items"] = _listing(
        _item("a", created_at=same_ts),
        _item("c", created_at=same_ts),
        _item("b", created_at=same_ts),
    )
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1")] == ["c", "b", "a"]


def test_list_items_requests_only_the_owners_collection() -> None:
    """Owner scoping happens by requesting the owner's own subcollection path."""
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {}
    FirestoreVaultIndex(transport).list_items("u1")
    method, path, _ = transport.calls[0]
    assert method == "GET"
    assert path == "vaults/u1/items"


def test_delete_item_always_returns_true() -> None:
    """Unlike SQLite, a Firestore DELETE has no rowcount to report on.

    Firestore's delete endpoint returns an empty 200 whether or not the
    document existed, so there is no cheap way to report whether anything
    was actually removed without an extra read first. Callers that need
    "did this exist" should check with get_item beforehand.
    """
    index = FirestoreVaultIndex(_StubTransport())
    assert index.delete_item("u1", "never-existed") is True


def test_put_solve_and_get_solve_round_trip_and_are_not_owner_scoped() -> None:
    transport = _StubTransport()
    index = FirestoreVaultIndex(transport)
    index.put_solve(_solve("s1", subject="physics"))
    method, path, body = transport.calls[0]
    assert method == "PATCH"
    assert path == "solves/s1"
    assert body is not None

    transport.responses["solves/s1"] = _doc(_solve("s1", subject="physics"))
    got = index.get_solve("s1")
    assert got is not None
    assert got.subject == "physics"


def test_get_solve_returns_none_when_absent() -> None:
    assert FirestoreVaultIndex(_StubTransport()).get_solve("missing") is None


def test_note_round_trips_and_is_owner_scoped() -> None:
    transport = _StubTransport()
    index = FirestoreVaultIndex(transport)
    index.put_note(_note("n1", owner_uid="u1", markdown="hi"))
    method, path, _ = transport.calls[0]
    assert method == "PATCH"
    assert path == "vaults/u1/notes/n1"

    transport.responses["vaults/u1/notes/n1"] = _doc(
        _note("n1", owner_uid="u1", markdown="hi")
    )
    got = index.get_note("u1", "n1")
    assert got is not None
    assert got.markdown == "hi"
    assert index.get_note("u2", "n1") is None


def test_list_notes_orders_newest_first() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/notes"] = _listing(
        _note("early", created_at="2026-08-16T09:00:00+00:00"),
        _note("late", created_at="2026-08-16T11:00:00+00:00"),
    )
    index = FirestoreVaultIndex(transport)
    assert [n.id for n in index.list_notes("u1")] == ["late", "early"]


def test_total_bytes_sums_committed_assets_including_locked() -> None:
    """Locked items still consume storage — quota accounting must count them.

    Matches SQLiteVaultIndex.total_bytes, which sums over every row
    regardless of privacy.
    """
    transport = _StubTransport()
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    transport.responses["vaults/u1/items"] = _listing(
        _item("open", cloudinary=asset),
        _item("shut", cloudinary=asset, privacy=Privacy.LOCKED),
    )
    index = FirestoreVaultIndex(transport)
    assert index.total_bytes("u1") == 2000


def test_total_bytes_ignores_items_with_no_committed_asset() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = _listing(_item("pending"))
    index = FirestoreVaultIndex(transport)
    assert index.total_bytes("u1") == 0


# --------------------------------------------------------------------------
# Malformed and unexpected transport responses.
# --------------------------------------------------------------------------


def test_get_item_returns_none_for_a_non_dict_response() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = "not a document"
    assert FirestoreVaultIndex(transport).get_item("u1", "i1") is None


def test_get_item_returns_none_when_doc_field_is_missing() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = {"fields": {"other": {"stringValue": "x"}}}
    assert FirestoreVaultIndex(transport).get_item("u1", "i1") is None


def test_get_item_returns_none_when_doc_is_not_valid_json() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = {
        "fields": {"doc": {"stringValue": "not even json"}}
    }
    assert FirestoreVaultIndex(transport).get_item("u1", "i1") is None


def test_get_item_returns_none_when_doc_is_valid_json_but_not_a_valid_item() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = {
        "fields": {"doc": {"stringValue": '{"not": "a valid item"}'}}
    }
    assert FirestoreVaultIndex(transport).get_item("u1", "i1") is None


def test_list_items_skips_a_malformed_document_and_keeps_the_rest() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {
        "documents": [
            _doc(_item("good")),
            {"fields": {"doc": {"stringValue": '{"not": "a valid item"}'}}},
        ]
    }
    index = FirestoreVaultIndex(transport)
    assert [i.id for i in index.list_items("u1")] == ["good"]


def test_list_items_handles_a_response_with_no_documents_key() -> None:
    """A dict with no `documents` key is a genuinely empty page — not an error.

    This is what a real empty Firestore collection actually returns.
    """
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {}
    assert FirestoreVaultIndex(transport).list_items("u1") == []


def test_malformed_row_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items/bad"] = {
        "fields": {"doc": {"stringValue": '{"not": "a valid item"}'}}
    }
    with caplog.at_level(logging.WARNING, logger="friday.vault.index"):
        assert FirestoreVaultIndex(transport).get_item("u1", "bad") is None
    assert any("bad" in record.message for record in caplog.records)


# --------------------------------------------------------------------------
# Pagination: a vault past Firestore's ~300-document page cap must not lose
# its tail.
# --------------------------------------------------------------------------


def test_list_items_follows_pagination_to_get_every_item() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {
        "documents": [_doc(_item("a"))],
        "nextPageToken": "page2",
    }
    transport.responses["vaults/u1/items?pageToken=page2"] = {
        "documents": [_doc(_item("b"))],
    }
    index = FirestoreVaultIndex(transport)
    assert {i.id for i in index.list_items("u1")} == {"a", "b"}

    paths = [path for _, path, _ in transport.calls]
    assert "vaults/u1/items" in paths
    assert "vaults/u1/items?pageToken=page2" in paths


def test_list_items_follows_a_chain_of_more_than_two_pages() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {
        "documents": [_doc(_item("a"))],
        "nextPageToken": "p2",
    }
    transport.responses["vaults/u1/items?pageToken=p2"] = {
        "documents": [_doc(_item("b"))],
        "nextPageToken": "p3",
    }
    transport.responses["vaults/u1/items?pageToken=p3"] = {
        "documents": [_doc(_item("c"))],
    }
    index = FirestoreVaultIndex(transport)
    assert {i.id for i in index.list_items("u1")} == {"a", "b", "c"}


def test_list_notes_follows_pagination_to_get_every_note() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/notes"] = {
        "documents": [_doc(_note("a"))],
        "nextPageToken": "page2",
    }
    transport.responses["vaults/u1/notes?pageToken=page2"] = {
        "documents": [_doc(_note("b"))],
    }
    index = FirestoreVaultIndex(transport)
    assert {n.id for n in index.list_notes("u1")} == {"a", "b"}


def test_total_bytes_sums_across_every_page_with_no_limit() -> None:
    """Critical: total_bytes must never undercount by capping the read.

    It feeds the quota check on every upload signature request — an
    undercount (whether from a page cutoff or a hardcoded limit) lets an
    upload through that should have been refused.
    """
    transport = _StubTransport()
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=500, resource_type="image"
    )
    transport.responses["vaults/u1/items"] = {
        "documents": [_doc(_item("a", cloudinary=asset))],
        "nextPageToken": "page2",
    }
    transport.responses["vaults/u1/items?pageToken=page2"] = {
        "documents": [_doc(_item("b", cloudinary=asset))],
    }
    index = FirestoreVaultIndex(transport)
    assert index.total_bytes("u1") == 1000


# --------------------------------------------------------------------------
# Outages must not look like an empty (or under-counted) vault.
# --------------------------------------------------------------------------


def test_list_items_raises_when_the_transport_returns_nothing_usable() -> None:
    """A `None` response cannot mean "empty" for a collection listing.

    A genuinely empty Firestore collection is still a dict (`{}`); `None`
    means the underlying transport didn't get a real answer. Reporting that
    as "no items" would let an outage silently mimic an empty vault, which
    combined with total_bytes would fail the upload quota check open.
    """
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(_StubTransport()).list_items("u1")


def test_list_items_raises_on_an_error_shaped_response() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {"error": {"code": 500, "message": "internal"}}
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).list_items("u1")


def test_total_bytes_raises_rather_than_undercounting_on_an_error_response() -> None:
    """The dangerous combination: quota accounting must not fail open.

    If total_bytes swallowed this and returned 0 instead, the quota check
    downstream would pass during a genuine Firestore outage.
    """
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {"error": {"code": 500, "message": "internal"}}
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).total_bytes("u1")


def test_list_items_raises_on_the_second_page_being_error_shaped() -> None:
    """The pagination loop must keep checking every page it fetches, not just the first."""
    transport = _StubTransport()
    transport.responses["vaults/u1/items"] = {
        "documents": [_doc(_item("a"))],
        "nextPageToken": "page2",
    }
    transport.responses["vaults/u1/items?pageToken=page2"] = {
        "error": {"code": 500, "message": "internal"}
    }
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).list_items("u1")


def test_get_item_raises_on_an_error_shaped_response() -> None:
    """A get must not report "not found" when Firestore actually errored."""
    transport = _StubTransport()
    transport.responses["vaults/u1/items/i1"] = {"error": {"code": 500, "message": "internal"}}
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).get_item("u1", "i1")


def test_get_solve_raises_on_an_error_shaped_response() -> None:
    transport = _StubTransport()
    transport.responses["solves/s1"] = {"error": {"code": 503, "message": "unavailable"}}
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).get_solve("s1")


def test_get_note_raises_on_an_error_shaped_response() -> None:
    transport = _StubTransport()
    transport.responses["vaults/u1/notes/n1"] = {"error": {"code": 500, "message": "internal"}}
    with pytest.raises(FirestoreIndexError):
        FirestoreVaultIndex(transport).get_note("u1", "n1")


def test_transport_exceptions_propagate_rather_than_being_swallowed() -> None:
    """Genuine transport failures (network errors, timeouts) must not be hidden.

    SQLite never raises on a query, so the two backends do diverge on this —
    but silently swallowing a Firestore outage into "empty vault" is worse:
    combined with total_bytes feeding the upload quota check, that would fail
    the quota guard open. The route layer is expected to catch this (and
    FirestoreIndexError) and return a 5xx.
    """

    class _RaisingTransport:
        def request(
            self, method: str, path: str, body: dict[str, Any] | None = None
        ) -> Any:
            raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        FirestoreVaultIndex(_RaisingTransport()).list_items("u1")

    with pytest.raises(ConnectionError):
        FirestoreVaultIndex(_RaisingTransport()).total_bytes("u1")

    with pytest.raises(ConnectionError):
        FirestoreVaultIndex(_RaisingTransport()).get_item("u1", "i1")
