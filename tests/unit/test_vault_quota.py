"""Unit tests for the storage budget and daily solve cap."""

from __future__ import annotations

import pytest

from friday.vault.firestore_index import FirestoreIndexError
from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import CaptureSource, CloudinaryAsset, Item
from friday.vault.quota import QuotaGuard

_GB = 1024**3


def _item(item_id: str, size: int) -> Item:
    return Item(
        id=item_id,
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        created_at="2026-08-16T10:00:00+00:00",
        cloudinary=CloudinaryAsset(
            public_id=item_id, version=1, format="jpg", bytes=size
        ),
    )


class _ExplodingIndex:
    """A stand-in whose ``total_bytes`` behaves like a Firestore outage.

    Only ``total_bytes`` is implemented: ``QuotaGuard`` only ever calls that
    one method, and this isn't type-checked as a `VaultIndex` (mypy's files
    list is ``src`` only), so a minimal duck-typed stub is enough here.
    """

    def total_bytes(self, owner_uid: str) -> int:
        raise FirestoreIndexError("firestore unreachable")


def test_reports_usage_and_headroom() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", _GB))
    usage = QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).usage("u1")
    assert usage.used_bytes == _GB
    assert usage.warning is False


def test_warns_past_eighty_percent() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", int(_GB * 21)))
    assert QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).usage("u1").warning is True


def test_refuses_upload_when_full() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", int(_GB * 25)))
    assert QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).may_upload("u1") is False


def test_solve_cap_counts_down_and_stops() -> None:
    guard = QuotaGuard(SQLiteVaultIndex(":memory:"), quota_gb=25.0, daily_solve_cap=2)
    assert guard.may_solve("2026-08-16") is True
    guard.record_solve("2026-08-16")
    guard.record_solve("2026-08-16")
    assert guard.may_solve("2026-08-16") is False
    assert guard.may_solve("2026-08-17") is True


def test_zero_cap_means_unlimited() -> None:
    guard = QuotaGuard(SQLiteVaultIndex(":memory:"), quota_gb=25.0, daily_solve_cap=0)
    for _ in range(500):
        guard.record_solve("2026-08-16")
    assert guard.may_solve("2026-08-16") is True


# ------------------------------------------------------- extra: outage safety


def test_usage_propagates_index_outage_instead_of_reporting_empty() -> None:
    """The whole point of ``FirestoreIndexError``: never mistaken for an empty vault."""
    guard = QuotaGuard(_ExplodingIndex(), quota_gb=25.0, daily_solve_cap=10)
    with pytest.raises(FirestoreIndexError):
        guard.usage("u1")


def test_may_upload_propagates_index_outage_rather_than_fail_open() -> None:
    """An outage must never be waved through as "plenty of room"."""
    guard = QuotaGuard(_ExplodingIndex(), quota_gb=25.0, daily_solve_cap=10)
    with pytest.raises(FirestoreIndexError):
        guard.may_upload("u1")


# ------------------------------------------------------------ extra: boundary


def test_warning_boundary_exact_eighty_percent_is_inclusive() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", int(_GB * 20)))  # exactly 80% of 25 GB
    usage = QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).usage("u1")
    assert usage.fraction == pytest.approx(0.8)
    assert usage.warning is True


def test_may_upload_true_one_byte_under_quota() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", int(_GB * 25) - 1))
    assert QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).may_upload("u1") is True


def test_may_upload_false_exactly_at_quota() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item("a", int(_GB * 25)))
    assert QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10).may_upload("u1") is False


# -------------------------------------------------------- extra: odd configs


def test_zero_quota_gb_refuses_every_upload() -> None:
    """Zero storage quota means zero room -- unlike the solve cap, 0 is not "unlimited" here.

    A misconfigured ``daily_solve_cap=0`` fails safe (it just means unlimited
    solves, a bounded nuisance). A misconfigured ``quota_gb=0`` failing the
    other way -- treating it as "unlimited storage" -- would be a much worse
    footgun, since storage cost is the budget this guard exists to bound. So
    the two zeros are deliberately asymmetric.
    """
    index = SQLiteVaultIndex(":memory:")
    assert QuotaGuard(index, quota_gb=0.0, daily_solve_cap=10).may_upload("u1") is False


def test_negative_quota_gb_refuses_every_upload_without_crashing() -> None:
    index = SQLiteVaultIndex(":memory:")
    assert QuotaGuard(index, quota_gb=-5.0, daily_solve_cap=10).may_upload("u1") is False


def test_solve_cap_is_per_process_and_keys_on_whatever_day_string_it_is_given() -> None:
    """No validation on ``day``: it's an opaque per-process bucket key, not a parsed date."""
    guard = QuotaGuard(SQLiteVaultIndex(":memory:"), quota_gb=25.0, daily_solve_cap=1)
    guard.record_solve("not-a-real-date")
    assert guard.may_solve("not-a-real-date") is False
    assert guard.may_solve("2026-08-16") is True
