"""Integration tests for the ``/schedules`` REST API (Tier 1 scheduler).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_SCHEDULER`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the reminders / RAG / studio API tests). No network,
no key.

Covered:
* Every ``/schedules`` surface is ``404`` when the flag is off.
* ``POST /schedules`` creates a trigger and computes its initial ``next_run``.
* ``GET /schedules`` lists triggers.
* ``POST /schedules/{id}/enable`` + ``/disable`` toggle the flag.
* ``POST /schedules/{id}/run`` fires the action now.
* ``DELETE /schedules/{id}`` removes a trigger.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_scheduler=True,
        # Keep reminders on too so the wired ``due_reminders`` action has its
        # shared store available for the run-now path.
        enable_reminders=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_scheduler=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose scheduler flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_schedules_disabled_create_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post(
            "/schedules",
            json={"name": "x", "kind": "interval", "spec": "60", "action": "noop"},
        )
    assert resp.status_code == 404


def test_schedules_disabled_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/schedules")
    assert resp.status_code == 404


def test_schedules_disabled_enable_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/schedules/1/enable")
    assert resp.status_code == 404


def test_schedules_disabled_run_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/schedules/1/run")
    assert resp.status_code == 404


def test_schedules_disabled_delete_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.delete("/schedules/1")
    assert resp.status_code == 404


def test_schedules_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), create is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post(
            "/schedules",
            json={"name": "x", "kind": "interval", "spec": "60", "action": "noop"},
        )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> CRUD + enable/disable + run-now
# --------------------------------------------------------------------------- #
def test_schedules_create_computes_next_run(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/schedules",
            json={
                "name": "nightly",
                "kind": "daily",
                "spec": "09:00",
                "action": "noop",
            },
        )
        assert created.status_code == 200
        body = created.json()
        assert body["name"] == "nightly"
        assert body["kind"] == "daily"
        assert body["enabled"] is True
        assert isinstance(body["id"], int)
        # The route computed an initial next_run from the spec (a daily HH:MM at
        # the next 09:00 occurrence). It is a parseable ISO timestamp at 09:00.
        assert body["next_run"] is not None
        parsed = datetime.fromisoformat(body["next_run"])
        assert (parsed.hour, parsed.minute, parsed.second) == (9, 0, 0)


def test_schedules_create_once_in_past_has_null_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/schedules",
            json={
                "name": "expired",
                "kind": "once",
                "spec": "2000-01-01T00:00:00+00:00",
                "action": "noop",
            },
        )
        assert created.status_code == 200
        assert created.json()["next_run"] is None


def test_schedules_create_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        client.post(
            "/schedules",
            json={"name": "a", "kind": "interval", "spec": "60", "action": "noop"},
        )
        client.post(
            "/schedules",
            json={"name": "b", "kind": "interval", "spec": "120", "action": "noop"},
        )
        listed = client.get("/schedules")
        assert listed.status_code == 200
        names = [t["name"] for t in listed.json()["schedules"]]
        assert names == ["a", "b"]
        assert listed.json()["count"] == 2


def test_schedules_create_invalid_kind_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post(
            "/schedules",
            json={"name": "x", "kind": "yearly", "spec": "60", "action": "noop"},
        )
    assert resp.status_code == 422


def test_schedules_enable_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/schedules",
            json={"name": "a", "kind": "interval", "spec": "60", "action": "noop"},
        )
        sid = created.json()["id"]

        disabled = client.post(f"/schedules/{sid}/disable")
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False

        enabled = client.post(f"/schedules/{sid}/enable")
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True

        # Reflected in the list view.
        listed = client.get("/schedules").json()["schedules"]
        assert listed[0]["enabled"] is True


def test_schedules_enable_unknown_id_is_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/schedules/999/enable")
    assert resp.status_code == 404


def test_schedules_run_now_fires_action(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/schedules",
            json={
                "name": "ping",
                "kind": "interval",
                "spec": "3600",
                "action": "noop",
            },
        )
        sid = created.json()["id"]

        ran = client.post(f"/schedules/{sid}/run")
        assert ran.status_code == 200
        assert ran.json()["ran"] is True
        assert ran.json()["id"] == sid


def test_schedules_run_unknown_id_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/schedules/999/run")
    assert resp.status_code == 404


def test_schedules_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/schedules",
            json={"name": "temp", "kind": "interval", "spec": "60", "action": "noop"},
        )
        sid = created.json()["id"]

        deleted = client.delete(f"/schedules/{sid}")
        assert deleted.status_code == 200
        assert deleted.json()["removed"] == 1

        assert client.get("/schedules").json()["schedules"] == []
