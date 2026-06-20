"""Integration tests for the /circle REST surface.

Offline: the lifespan builds the app; each test sets a token->uid map on
``app.state`` (the identity seam Firebase replaces later) and drives the full
flow — create group, invite, accept, list members, set + read status — plus the
flag-off (404) and unauthenticated (401) paths. Services are built lazily on
in-memory stores by the routes themselves.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings

ADMIN = {"Authorization": "Bearer tok-a"}
FRIEND = {"Authorization": "Bearer tok-b"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_ENABLE_CIRCLE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client_with_identities() -> TestClient:
    client = TestClient(create_app())
    client.__enter__()
    client.app.state.siri_identities = {"tok-a": "u-a", "tok-b": "u-b"}
    return client


def test_full_group_invite_status_flow() -> None:
    client = _client_with_identities()
    try:
        # Admin creates a group.
        created = client.post(
            "/circle/groups", json={"name": "Us", "display_name": "Me"}, headers=ADMIN
        )
        assert created.status_code == 200
        gid = created.json()["id"]

        # Admin mints an invite; the friend accepts it.
        code = client.post(
            f"/circle/groups/{gid}/invites", json={}, headers=ADMIN
        ).json()["code"]
        accepted = client.post(
            f"/circle/invites/{code}/accept",
            json={"display_name": "Bestie", "tz": "America/New_York"},
            headers=FRIEND,
        )
        assert accepted.status_code == 200
        assert accepted.json()["role"] == "member"

        # Both are members now.
        members = client.get(
            f"/circle/groups/{gid}/members", headers=ADMIN
        ).json()["members"]
        assert {m["uid"] for m in members} == {"u-a", "u-b"}

        # Friend sets status; admin reads it (consent via shared group).
        assert (
            client.put(
                "/circle/status", json={"text": "having lunch"}, headers=FRIEND
            ).status_code
            == 200
        )
        spoken = client.get("/circle/status/u-b", headers=ADMIN).json()["speak"]
        assert "Bestie" in spoken
        assert "having lunch" in spoken
    finally:
        client.__exit__(None, None, None)


def test_non_admin_cannot_invite() -> None:
    client = _client_with_identities()
    try:
        gid = client.post(
            "/circle/groups", json={"name": "Us"}, headers=ADMIN
        ).json()["id"]
        # tok-b is not a member/admin of this group -> 403.
        resp = client.post(f"/circle/groups/{gid}/invites", json={}, headers=FRIEND)
        assert resp.status_code == 403
    finally:
        client.__exit__(None, None, None)


def test_unauthenticated_is_401() -> None:
    client = _client_with_identities()
    try:
        resp = client.post("/circle/groups", json={"name": "Us"})
        assert resp.status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_disabled_flag_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_CIRCLE", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        resp = client.post("/circle/groups", json={"name": "Us"}, headers=ADMIN)
    assert resp.status_code == 404
