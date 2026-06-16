# © Lakshya Badjatya — Author
"""Integration tests for role-based access control (the /admin gate)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _rbac_client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "true")
    monkeypatch.setenv("FRIDAY_API_KEYS", "owner-key, member-key")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_RBAC", "true")
        monkeypatch.setenv("FRIDAY_API_ROLES", "owner-key=owner, member-key=member")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_RBAC", raising=False)
        monkeypatch.delenv("FRIDAY_API_ROLES", raising=False)
    get_settings.cache_clear()
    return TestClient(create_app())


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_member_denied_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    with _rbac_client(monkeypatch, enabled=True) as client:
        resp = client.get("/admin/audit", headers=_auth("member-key"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "forbidden"


def test_owner_allowed_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    with _rbac_client(monkeypatch, enabled=True) as client:
        resp = client.get("/admin/audit", headers=_auth("owner-key"))
    assert resp.status_code != 403  # owner passes RBAC (200)


def test_member_allowed_non_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    # RBAC only gates /admin; a member key still reaches an ordinary route.
    with _rbac_client(monkeypatch, enabled=True) as client:
        resp = client.get("/roster", headers=_auth("member-key"))
    assert resp.status_code == 200


def test_rbac_off_leaves_admin_open_to_any_valid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _rbac_client(monkeypatch, enabled=False) as client:
        resp = client.get("/admin/audit", headers=_auth("member-key"))
    assert resp.status_code != 403  # no RBAC -> a valid key is not forbidden
