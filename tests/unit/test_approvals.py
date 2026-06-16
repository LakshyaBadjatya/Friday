# © Lakshya Badjatya — Author
"""Unit tests for the approval-workflow state machine."""

from __future__ import annotations

import pytest

from friday.security.approvals import ApprovalStore


def test_create_and_approve() -> None:
    store = ApprovalStore()
    req = store.create("wire $500", request_id="a1", now=0.0)
    assert req.status == "pending"
    assert store.is_approved("a1", now=1.0) is False
    decided = store.approve("a1", now=1.0)
    assert decided.status == "approved"
    assert decided.decided_ts == 1.0
    assert store.is_approved("a1", now=2.0) is True


def test_deny_blocks_approval() -> None:
    store = ApprovalStore()
    store.create("delete everything", request_id="a1", now=0.0)
    store.deny("a1", now=1.0)
    assert store.is_approved("a1", now=2.0) is False


def test_cannot_decide_twice() -> None:
    store = ApprovalStore()
    store.create("x", request_id="a1", now=0.0)
    store.approve("a1", now=1.0)
    with pytest.raises(ValueError, match="already approved"):
        store.deny("a1", now=2.0)


def test_duplicate_id_rejected() -> None:
    store = ApprovalStore()
    store.create("x", request_id="a1", now=0.0)
    with pytest.raises(ValueError, match="already exists"):
        store.create("y", request_id="a1", now=0.0)


def test_unknown_id_raises_on_decide() -> None:
    store = ApprovalStore()
    with pytest.raises(KeyError):
        store.approve("ghost", now=0.0)
    assert store.is_approved("ghost", now=0.0) is False


def test_ttl_expiry_blocks_late_approval() -> None:
    store = ApprovalStore()
    store.create("x", request_id="a1", now=0.0, ttl_seconds=10.0)
    # Past the TTL, the pending request has expired and cannot be approved.
    with pytest.raises(ValueError, match="expired"):
        store.approve("a1", now=10.0)
    assert store.is_approved("a1", now=11.0) is False
    assert store.get("a1").status == "expired"  # type: ignore[union-attr]


def test_pending_lists_only_live_requests() -> None:
    store = ApprovalStore()
    store.create("live", request_id="a1", now=0.0)
    store.create("expiring", request_id="a2", now=0.0, ttl_seconds=5.0)
    store.create("decided", request_id="a3", now=0.0)
    store.approve("a3", now=1.0)
    pending = store.pending(now=6.0)  # a2 has expired by now
    assert [r.id for r in pending] == ["a1"]
