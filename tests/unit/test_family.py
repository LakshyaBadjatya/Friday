"""Unit tests for the consent-enforced family-sharing feature (build-spec §18).

These tests exercise the store (:class:`~friday.family.store.SQLiteFamilyStore`)
and the service (:class:`~friday.family.service.FamilyService`) directly —
offline, with an injected clock so timestamps are deterministic, and a
``":memory:"`` SQLite database so nothing touches the real data path.

The NON-NEGOTIABLE consent guardrails (build-spec §18) verified here:

* (1) A participant can only be added by THEMSELVES (``self_opt_in=True``
  required); attempting to add someone from another account is rejected.
* (3) A unilateral revoke stops sharing INSTANTLY (the sharing edge is gone, so
  a subsequent view by the revoked viewer is denied).
* (4) A view is recorded, and the VIEWED participant can see who viewed them.
* The DEFAULT share is geofence STATUS (home/work/away), never raw coordinates;
  raw location requires an explicit per-viewer opt-in.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from friday.family.service import FamilyService, FamilyShareError
from friday.family.store import Participant, SQLiteFamilyStore


def _clock() -> datetime:
    """A fixed, deterministic clock for stamping views/opt-ins."""
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


def _store() -> SQLiteFamilyStore:
    return SQLiteFamilyStore(":memory:", clock=_clock)


def _service() -> FamilyService:
    return FamilyService(_store())


# --------------------------------------------------------------------------- #
# Store: opt-in is a self-opt-in, defaults are safe
# --------------------------------------------------------------------------- #
def test_optin_creates_self_opted_participant() -> None:
    store = _store()
    p = store.opt_in("alice")
    assert isinstance(p, Participant)
    assert p.name == "alice"
    assert p.self_opted_in is True
    assert p.status == "away"  # geofence default, never raw coordinates
    assert p.sharing_with == []
    assert isinstance(p.id, int)


def test_optin_is_idempotent_by_name() -> None:
    store = _store()
    first = store.opt_in("alice")
    second = store.opt_in("alice")
    assert first.id == second.id
    assert len(store.list_participants()) == 1


def test_get_unknown_participant_is_none() -> None:
    store = _store()
    assert store.get("nobody") is None


# --------------------------------------------------------------------------- #
# Guardrail (1): a participant can only be added by themselves
# --------------------------------------------------------------------------- #
def test_optin_requires_self_opt_in_true() -> None:
    """Adding someone from another account (self_opt_in=False) is rejected."""
    service = _service()
    with pytest.raises(FamilyShareError):
        service.opt_in("bob", self_opt_in=False)
    # Nothing was persisted by the rejected call.
    assert service.store.get("bob") is None


def test_optin_with_self_opt_in_true_succeeds() -> None:
    service = _service()
    p = service.opt_in("bob", self_opt_in=True)
    assert p.self_opted_in is True
    assert service.store.get("bob") is not None


# --------------------------------------------------------------------------- #
# Sharing requires the sharer to have opted in
# --------------------------------------------------------------------------- #
def test_share_requires_owner_opted_in() -> None:
    service = _service()
    # "ghost" never opted in -> cannot share with anyone.
    with pytest.raises(FamilyShareError):
        service.share("ghost", "alice")


def test_share_adds_viewer_to_sharing_with() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    updated = service.share("alice", "bob")
    assert "bob" in updated.sharing_with


def test_share_is_idempotent() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob")
    updated = service.share("alice", "bob")
    assert updated.sharing_with.count("bob") == 1


# --------------------------------------------------------------------------- #
# Default share is geofence STATUS, not raw coordinates
# --------------------------------------------------------------------------- #
def test_default_view_is_geofence_status_not_coordinates() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob")
    view = service.view("alice", viewer="bob")
    # The default share is the coarse geofence status, never raw coordinates.
    assert view["status"] in {"home", "work", "away"}
    assert "latitude" not in view
    assert "longitude" not in view
    assert view["precision"] == "status"


def test_raw_location_requires_explicit_per_viewer_optin() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob")
    # Without the explicit raw opt-in, the view is status-only.
    coarse = service.view("alice", viewer="bob")
    assert "latitude" not in coarse

    # Alice explicitly opts bob in to RAW location for her own record.
    service.share("alice", "bob", raw_location=True)
    fine = service.view("alice", viewer="bob")
    assert fine["precision"] == "raw"
    assert "latitude" in fine and "longitude" in fine


def test_raw_optin_is_per_viewer_not_global() -> None:
    """Granting raw to one viewer does not leak raw to another viewer."""
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.opt_in("carol", self_opt_in=True)
    service.share("alice", "bob", raw_location=True)
    service.share("alice", "carol")  # carol gets default (status) only

    assert service.view("alice", viewer="bob")["precision"] == "raw"
    assert service.view("alice", viewer="carol")["precision"] == "status"


# --------------------------------------------------------------------------- #
# Guardrail (4): a view is recorded; the viewed participant sees who viewed them
# --------------------------------------------------------------------------- #
def test_view_is_recorded_and_visible_to_viewed() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob")
    service.view("alice", viewer="bob")

    views = service.views_of("alice")
    assert len(views) == 1
    assert views[0]["viewer"] == "bob"
    assert views[0]["viewed"] == "alice"
    assert views[0]["at"] == _clock().isoformat()


def test_multiple_views_recorded_most_recent_first() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.opt_in("carol", self_opt_in=True)
    service.share("alice", "bob")
    service.share("alice", "carol")
    service.view("alice", viewer="bob")
    service.view("alice", viewer="carol")

    viewers = [v["viewer"] for v in service.views_of("alice")]
    assert viewers == ["carol", "bob"]


# --------------------------------------------------------------------------- #
# A viewer who is not shared-with cannot view
# --------------------------------------------------------------------------- #
def test_view_denied_when_not_shared_with() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("eve", self_opt_in=True)
    # alice never shared with eve.
    with pytest.raises(FamilyShareError):
        service.view("alice", viewer="eve")
    # The denied attempt records no view.
    assert service.views_of("alice") == []


# --------------------------------------------------------------------------- #
# Guardrail (3): a unilateral revoke stops sharing INSTANTLY
# --------------------------------------------------------------------------- #
def test_revoke_stops_sharing_instantly() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob")
    # bob can view before the revoke.
    service.view("alice", viewer="bob")

    updated = service.revoke("alice", "bob")
    assert "bob" not in updated.sharing_with

    # Immediately after the revoke bob can no longer view.
    with pytest.raises(FamilyShareError):
        service.view("alice", viewer="bob")


def test_revoke_also_drops_raw_optin() -> None:
    """Revoking a viewer also removes any raw-location grant for that viewer."""
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    service.opt_in("bob", self_opt_in=True)
    service.share("alice", "bob", raw_location=True)
    service.revoke("alice", "bob")
    # Re-share at the default level: bob must be status-only again, not raw.
    service.share("alice", "bob")
    assert service.view("alice", viewer="bob")["precision"] == "status"


def test_revoke_unknown_edge_is_noop() -> None:
    service = _service()
    service.opt_in("alice", self_opt_in=True)
    # Revoking a viewer that was never shared with is a harmless no-op.
    updated = service.revoke("alice", "bob")
    assert updated.sharing_with == []


def test_revoke_requires_owner_opted_in() -> None:
    service = _service()
    with pytest.raises(FamilyShareError):
        service.revoke("ghost", "bob")


# --------------------------------------------------------------------------- #
# Status update stays geofence-coarse
# --------------------------------------------------------------------------- #
def test_set_status_updates_geofence_only() -> None:
    store = _store()
    store.opt_in("alice")
    store.set_status("alice", "home")
    p = store.get("alice")
    assert p is not None
    assert p.status == "home"


def test_set_status_rejects_unknown_geofence() -> None:
    store = _store()
    store.opt_in("alice")
    with pytest.raises(ValueError):
        store.set_status("alice", "spaceship")  # not a known geofence label
