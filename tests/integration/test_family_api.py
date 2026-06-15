"""Integration tests for the flagged ``/family`` surface (build-spec §18).

Consent-enforced family sharing: every member opts THEMSELVES in, shares with a
named viewer, can unilaterally revoke instantly, and the default share is the
coarse geofence status (home/work/away) — never raw coordinates. A view is
recorded and the viewed member can see who viewed them.

These tests mount :data:`friday.family.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The flag (``enable_family_sharing``) and the
store are read/built lazily inside the route from
:func:`~friday.config.get_settings`, so every test gets an isolated in-memory
store.

Covered:
* Every ``/family`` surface is ``404`` when the flag is off.
* ``POST /family/optin`` opts a member in (self-opt-in).
* ``POST /family/share`` adds a viewer (only after opt-in).
* ``GET  /family/status/{name}`` records a view and returns geofence STATUS, not
  coordinates.
* ``GET  /family/views/{name}`` shows who viewed the member.
* ``POST /family/revoke`` stops sharing instantly (a later view -> 403).
* Adding a participant from another account (self_opt_in=False) is rejected.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_family as routes_family
from friday.config import Settings
from friday.family import router as family_router


def _app() -> FastAPI:
    """A fresh app with ONLY the family router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(family_router)
    return app


def _enabled_settings() -> Settings:
    return Settings(
        _env_file=None, enable_family_sharing=True, memory_db_path=":memory:"
    )


def _disabled_settings() -> Settings:
    return Settings(
        _env_file=None, enable_family_sharing=False, memory_db_path=":memory:"
    )


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the family flag on and reset the lazy per-process store."""
    monkeypatch.setattr(routes_family, "get_settings", _enabled_settings)
    routes_family.reset_store()


@pytest.fixture
def disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_family, "get_settings", _disabled_settings)
    routes_family.reset_store()


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_family_disabled_optin_is_404(disabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post("/family/optin", json={"name": "alice"})
    assert resp.status_code == 404


def test_family_disabled_share_is_404(disabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post("/family/share", json={"owner": "alice", "viewer": "bob"})
    assert resp.status_code == 404


def test_family_disabled_status_is_404(disabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.get("/family/status/alice?viewer=bob")
    assert resp.status_code == 404


def test_family_disabled_views_is_404(disabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.get("/family/views/alice")
    assert resp.status_code == 404


def test_family_disabled_revoke_is_404(disabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post("/family/revoke", json={"owner": "alice", "viewer": "bob"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Opt-in (self-opt-in) — and guardrail (1)
# --------------------------------------------------------------------------- #
def test_optin_self_opts_member_in(enabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post("/family/optin", json={"name": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "alice"
    assert body["self_opted_in"] is True
    assert body["status"] in {"home", "work", "away"}
    assert body["sharing_with"] == []


def test_optin_bad_body_is_422(enabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post("/family/optin", json={})
    assert resp.status_code == 422


def test_optin_from_another_account_is_rejected(enabled: None) -> None:
    """Guardrail (1): adding someone with self_opt_in=False is rejected (403)."""
    with TestClient(_app()) as client:
        resp = client.post(
            "/family/optin", json={"name": "bob", "self_opt_in": False}
        )
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Share requires opt-in; default view is geofence STATUS not coordinates
# --------------------------------------------------------------------------- #
def test_share_before_optin_is_rejected(enabled: None) -> None:
    with TestClient(_app()) as client:
        resp = client.post(
            "/family/share", json={"owner": "ghost", "viewer": "bob"}
        )
    assert resp.status_code == 403


def test_share_then_status_returns_geofence_not_coordinates(enabled: None) -> None:
    with TestClient(_app()) as client:
        client.post("/family/optin", json={"name": "alice"})
        client.post("/family/optin", json={"name": "bob"})
        shared = client.post(
            "/family/share", json={"owner": "alice", "viewer": "bob"}
        )
        assert shared.status_code == 200
        assert "bob" in shared.json()["sharing_with"]

        view = client.get("/family/status/alice?viewer=bob")
    assert view.status_code == 200
    body = view.json()
    # Default share is the coarse geofence status, never raw coordinates.
    assert body["status"] in {"home", "work", "away"}
    assert body["precision"] == "status"
    assert "latitude" not in body
    assert "longitude" not in body


def test_status_view_by_unshared_viewer_is_403(enabled: None) -> None:
    with TestClient(_app()) as client:
        client.post("/family/optin", json={"name": "alice"})
        client.post("/family/optin", json={"name": "eve"})
        view = client.get("/family/status/alice?viewer=eve")
    assert view.status_code == 403


def test_raw_location_share_returns_coordinates(enabled: None) -> None:
    with TestClient(_app()) as client:
        client.post("/family/optin", json={"name": "alice"})
        client.post("/family/optin", json={"name": "bob"})
        client.post(
            "/family/share",
            json={"owner": "alice", "viewer": "bob", "raw_location": True},
        )
        view = client.get("/family/status/alice?viewer=bob")
    assert view.status_code == 200
    body = view.json()
    assert body["precision"] == "raw"
    assert "latitude" in body and "longitude" in body


# --------------------------------------------------------------------------- #
# Guardrail (4): a view is recorded + visible to the viewed member
# --------------------------------------------------------------------------- #
def test_view_is_recorded_and_visible(enabled: None) -> None:
    with TestClient(_app()) as client:
        client.post("/family/optin", json={"name": "alice"})
        client.post("/family/optin", json={"name": "bob"})
        client.post("/family/share", json={"owner": "alice", "viewer": "bob"})
        client.get("/family/status/alice?viewer=bob")

        views = client.get("/family/views/alice")
    assert views.status_code == 200
    body = views.json()
    assert body["count"] == 1
    assert body["views"][0]["viewer"] == "bob"
    assert body["views"][0]["viewed"] == "alice"


# --------------------------------------------------------------------------- #
# Guardrail (3): a unilateral revoke stops sharing instantly
# --------------------------------------------------------------------------- #
def test_revoke_stops_sharing_instantly(enabled: None) -> None:
    with TestClient(_app()) as client:
        client.post("/family/optin", json={"name": "alice"})
        client.post("/family/optin", json={"name": "bob"})
        client.post("/family/share", json={"owner": "alice", "viewer": "bob"})
        # bob can view before the revoke.
        before = client.get("/family/status/alice?viewer=bob")
        assert before.status_code == 200

        revoked = client.post(
            "/family/revoke", json={"owner": "alice", "viewer": "bob"}
        )
        assert revoked.status_code == 200
        assert "bob" not in revoked.json()["sharing_with"]

        # Immediately after the revoke, bob can no longer view.
        after = client.get("/family/status/alice?viewer=bob")
    assert after.status_code == 403
