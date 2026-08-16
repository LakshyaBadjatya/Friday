"""Unit tests for the SQLite vault index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import (
    CaptureSource,
    Classification,
    CloudinaryAsset,
    Item,
    ItemStatus,
    Note,
    Privacy,
    Solve,
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


def _solve(solve_id: str = "s1", **kw: object) -> Solve:
    base: dict[str, object] = {
        "id": solve_id,
        "item_ids": ["i1"],
        "created_at": "2026-08-16T10:00:00+00:00",
    }
    base.update(kw)
    return Solve.model_validate(base)


def test_put_then_get_round_trips() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())
    got = index.get_item("u1", "i1")
    assert got is not None
    assert got.owner_uid == "u1"
    assert got.source is CaptureSource.CAMERA


def test_get_refuses_another_owner() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())
    assert index.get_item("u2", "i1") is None


def test_list_filters_by_subject() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", classification=Classification(subject="physics")))
    index.put_item(_item("b", classification=Classification(subject="chemistry")))
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["a"]


def test_list_hides_locked_items_until_asked() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("open"))
    index.put_item(_item("shut", privacy=Privacy.LOCKED))
    assert [i.id for i in index.list_items("u1")] == ["open"]
    assert {i.id for i in index.list_items("u1", include_locked=True)} == {"open", "shut"}


def test_put_is_an_upsert() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())
    index.put_item(_item(status=ItemStatus.READY, ocr_text="v = ir"))
    got = index.get_item("u1", "i1")
    assert got is not None
    assert got.status is ItemStatus.READY
    assert got.ocr_text == "v = ir"
    assert len(index.list_items("u1")) == 1


def test_delete_removes_it() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())
    assert index.delete_item("u1", "i1") is True
    assert index.get_item("u1", "i1") is None
    assert index.delete_item("u1", "i1") is False


def test_total_bytes_sums_committed_assets() -> None:
    index = SQLiteVaultIndex(":memory:")
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    index.put_item(_item("a", cloudinary=asset))
    index.put_item(_item("b", cloudinary=asset))
    assert index.total_bytes("u1") == 2000


# --------------------------------------------------------------------------
# Extra coverage beyond the required minimum.
# --------------------------------------------------------------------------


def test_list_hides_locked_items_when_combined_with_subject_filter() -> None:
    """A locked item matching the subject filter must still be hidden.

    Guards against a `WHERE` clause built by string concatenation that drops
    the privacy condition once another filter is added.
    """
    index = SQLiteVaultIndex(":memory:")
    index.put_item(
        _item(
            "open",
            classification=Classification(subject="physics"),
        )
    )
    index.put_item(
        _item(
            "shut",
            classification=Classification(subject="physics"),
            privacy=Privacy.LOCKED,
        )
    )
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["open"]
    assert {i.id for i in index.list_items("u1", subject="physics", include_locked=True)} == {
        "open",
        "shut",
    }


def test_list_hides_locked_items_when_combined_with_kind_filter() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("open", classification=Classification(kind="homework")))
    index.put_item(
        _item(
            "shut",
            classification=Classification(kind="homework"),
            privacy=Privacy.LOCKED,
        )
    )
    assert [i.id for i in index.list_items("u1", kind="homework")] == ["open"]


def test_list_items_scoped_to_owner() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", owner_uid="u1"))
    index.put_item(_item("b", owner_uid="u2"))
    assert [i.id for i in index.list_items("u1")] == ["a"]
    assert [i.id for i in index.list_items("u2")] == ["b"]


def test_delete_item_refuses_another_owner() -> None:
    """Owner A must not be able to delete owner B's item by id alone."""
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("i1", owner_uid="u2"))
    assert index.delete_item("u1", "i1") is False
    assert index.get_item("u2", "i1") is not None


def test_total_bytes_scoped_to_owner() -> None:
    index = SQLiteVaultIndex(":memory:")
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    index.put_item(_item("a", owner_uid="u1", cloudinary=asset))
    index.put_item(_item("b", owner_uid="u2", cloudinary=asset))
    assert index.total_bytes("u1") == 1000
    assert index.total_bytes("u2") == 1000


def test_upsert_updates_promoted_columns_not_just_json() -> None:
    """Re-putting with a new subject/privacy must move the promoted columns.

    If only the JSON blob were updated, `get_item` would look right while
    `list_items` filtering on the stale column silently returned wrong
    results.
    """
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item(classification=Classification(subject="physics")))
    assert [i.id for i in index.list_items("u1", subject="physics")] == ["i1"]
    assert index.list_items("u1", subject="chemistry") == []

    index.put_item(_item(classification=Classification(subject="chemistry")))
    assert index.list_items("u1", subject="physics") == []
    assert [i.id for i in index.list_items("u1", subject="chemistry")] == ["i1"]

    index.put_item(_item(privacy=Privacy.LOCKED))
    assert index.list_items("u1") == []
    assert [i.id for i in index.list_items("u1", include_locked=True)] == ["i1"]


def test_list_items_orders_newest_first_and_respects_limit() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("early", created_at="2026-08-16T09:00:00+00:00"))
    index.put_item(_item("late", created_at="2026-08-16T11:00:00+00:00"))
    index.put_item(_item("mid", created_at="2026-08-16T10:00:00+00:00"))
    assert [i.id for i in index.list_items("u1")] == ["late", "mid", "early"]
    assert [i.id for i in index.list_items("u1", limit=2)] == ["late", "mid"]


def test_list_items_breaks_created_at_ties_by_id_descending() -> None:
    """Items sharing a timestamp must still sort deterministically.

    Without a secondary key, rows with equal `created_at` have no guaranteed
    order in SQL, and the Firestore backend (which sorts client-side over a
    collection returned in unspecified order) could legitimately disagree
    with SQLite on the tie order — a divergence a real caller would notice as
    a vault list that reshuffles between backends.
    """
    index = SQLiteVaultIndex(":memory:")
    same_ts = "2026-08-16T10:00:00+00:00"
    index.put_item(_item("a", created_at=same_ts))
    index.put_item(_item("c", created_at=same_ts))
    index.put_item(_item("b", created_at=same_ts))
    assert [i.id for i in index.list_items("u1")] == ["c", "b", "a"]


def test_file_backed_index_persists_across_instances(tmp_path: Path) -> None:
    """The production configuration: two separate instances, same file path."""
    db_path = str(tmp_path / "vault.db")
    first = SQLiteVaultIndex(db_path)
    first.put_item(_item())

    second = SQLiteVaultIndex(db_path)
    got = second.get_item("u1", "i1")
    assert got is not None
    assert got.owner_uid == "u1"

    second.put_item(_item(status=ItemStatus.READY))
    third = SQLiteVaultIndex(db_path)
    got2 = third.get_item("u1", "i1")
    assert got2 is not None
    assert got2.status is ItemStatus.READY


def test_file_backed_connections_are_actually_closed(tmp_path: Path) -> None:
    """A file-backed index must close its per-call connections.

    Proves the real invariant, not just that data round-trips: a held-open
    connection would still pass the persistence test above. ``sqlite3.Connection``
    is a C type and does not allow patching its ``close`` method (attribute is
    read-only), so this asserts the observable effect instead — a connection
    handed back by ``_close`` is genuinely unusable afterwards for a file path,
    while the shared ``:memory:`` connection stays open and usable across many
    calls to ``_close``.
    """
    db_path = str(tmp_path / "vault.db")
    file_index = SQLiteVaultIndex(db_path)

    conn = file_index._conn()  # noqa: SLF001 - exercising the connection lifecycle directly
    file_index._close(conn)  # noqa: SLF001
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    # The public API must still work after that connection was closed — each
    # call opens (and closes) its own.
    file_index.put_item(_item())
    assert file_index.get_item("u1", "i1") is not None

    memory_index = SQLiteVaultIndex(":memory:")
    shared_conn = memory_index._conn()  # noqa: SLF001
    for _ in range(3):
        memory_index._close(shared_conn)  # noqa: SLF001
    # Never actually closed: still usable after repeated _close() calls.
    shared_conn.execute("SELECT 1")
    memory_index.put_item(_item())
    assert memory_index.get_item("u1", "i1") is not None


def test_note_round_trips_and_is_owner_scoped() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_note(_note("n1", owner_uid="u1", markdown="hi"))
    index.put_note(_note("n2", owner_uid="u2", markdown="bye"))

    got = index.get_note("u1", "n1")
    assert got is not None
    assert got.markdown == "hi"
    assert index.get_note("u2", "n1") is None
    assert [n.id for n in index.list_notes("u1")] == ["n1"]
    assert [n.id for n in index.list_notes("u2")] == ["n2"]


def test_note_list_orders_newest_first() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_note(_note("early", created_at="2026-08-16T09:00:00+00:00"))
    index.put_note(_note("late", created_at="2026-08-16T11:00:00+00:00"))
    assert [n.id for n in index.list_notes("u1")] == ["late", "early"]


def test_solve_round_trips_and_missing_returns_none() -> None:
    index = SQLiteVaultIndex(":memory:")
    assert index.get_solve("missing") is None
    index.put_solve(_solve("s1", subject="physics"))
    got = index.get_solve("s1")
    assert got is not None
    assert got.subject == "physics"


def test_solve_put_is_an_upsert() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_solve(_solve("s1", subject="physics"))
    index.put_solve(_solve("s1", subject="chemistry"))
    got = index.get_solve("s1")
    assert got is not None
    assert got.subject == "chemistry"


# --------------------------------------------------------------------------
# Malformed rows: model evolution must not take the whole vault down.
# --------------------------------------------------------------------------


def _corrupt_item_doc(index: SQLiteVaultIndex, owner_uid: str, item_id: str) -> None:
    """Overwrite a stored item's ``doc`` with JSON that fails ``Item`` validation."""
    conn = index._conn()  # noqa: SLF001 - reaching in deliberately to simulate drift
    conn.execute(
        "UPDATE items SET doc = ? WHERE owner_uid = ? AND id = ?",
        ('{"not": "a valid item"}', owner_uid, item_id),
    )
    conn.commit()


def _corrupt_note_doc(index: SQLiteVaultIndex, owner_uid: str, note_id: str) -> None:
    conn = index._conn()  # noqa: SLF001
    conn.execute(
        "UPDATE notes SET doc = ? WHERE owner_uid = ? AND id = ?",
        ("not even json", owner_uid, note_id),
    )
    conn.commit()


def _corrupt_solve_doc(index: SQLiteVaultIndex, solve_id: str) -> None:
    conn = index._conn()  # noqa: SLF001
    conn.execute(
        "UPDATE solves SET doc = ? WHERE id = ?",
        ('{"not": "a valid solve"}', solve_id),
    )
    conn.commit()


def test_list_items_skips_a_malformed_row_and_keeps_the_rest() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("good"))
    index.put_item(_item("bad"))
    _corrupt_item_doc(index, "u1", "bad")

    assert [i.id for i in index.list_items("u1")] == ["good"]


def test_get_item_returns_none_for_a_malformed_row() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("bad"))
    _corrupt_item_doc(index, "u1", "bad")

    assert index.get_item("u1", "bad") is None


def test_list_notes_skips_a_malformed_row_and_keeps_the_rest() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_note(_note("good"))
    index.put_note(_note("bad"))
    _corrupt_note_doc(index, "u1", "bad")

    assert [n.id for n in index.list_notes("u1")] == ["good"]


def test_get_note_returns_none_for_a_malformed_row() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_note(_note("bad"))
    _corrupt_note_doc(index, "u1", "bad")

    assert index.get_note("u1", "bad") is None


def test_get_solve_returns_none_for_a_malformed_row() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_solve(_solve("bad"))
    _corrupt_solve_doc(index, "bad")

    assert index.get_solve("bad") is None


def test_malformed_row_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("bad"))
    _corrupt_item_doc(index, "u1", "bad")

    with caplog.at_level(logging.WARNING, logger="friday.vault.index"):
        assert index.get_item("u1", "bad") is None

    assert any("bad" in record.message for record in caplog.records)


# --------------------------------------------------------------------------
# The promoted `bytes` column: quota accounting must not re-parse every row.
# --------------------------------------------------------------------------


def test_total_bytes_uses_the_promoted_column_not_json_parsing() -> None:
    """Corrupting the JSON blob must not break `total_bytes`.

    If `total_bytes` still parsed every row's JSON to sum `cloudinary.bytes`,
    a single malformed doc would make quota accounting blow up. With bytes
    promoted to a real column, the sum is computed in SQL and never touches
    `doc` at all.
    """
    index = SQLiteVaultIndex(":memory:")
    asset = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=1000, resource_type="image"
    )
    index.put_item(_item("a", cloudinary=asset))
    index.put_item(_item("bad", cloudinary=asset))
    _corrupt_item_doc(index, "u1", "bad")

    assert index.total_bytes("u1") == 2000


def test_total_bytes_upsert_updates_the_promoted_column() -> None:
    """Re-putting an item with a different asset must move the stored `bytes`.

    A stale `bytes` value would silently corrupt quota accounting the same
    way a stale `subject` column would corrupt subject filtering.
    """
    index = SQLiteVaultIndex(":memory:")
    small = CloudinaryAsset(
        public_id="p", version=1, format="jpg", bytes=100, resource_type="image"
    )
    big = CloudinaryAsset(
        public_id="p", version=2, format="jpg", bytes=9000, resource_type="image"
    )
    index.put_item(_item("a", cloudinary=small))
    assert index.total_bytes("u1") == 100

    index.put_item(_item("a", cloudinary=big))
    assert index.total_bytes("u1") == 9000

    index.put_item(_item("a", cloudinary=None))
    assert index.total_bytes("u1") == 0
