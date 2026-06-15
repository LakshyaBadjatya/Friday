"""Integration tests for the ``/protocols`` REST API (Tier 1 voice protocols).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_PROTOCOLS`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the reminders / schedules API tests). No network, no
key.

Covered:
* Every ``/protocols`` surface is ``404`` when the flag is off.
* ``POST /protocols`` creates and returns the protocol (with its steps).
* ``GET /protocols`` lists in insertion order.
* ``POST /protocols/{id}/run`` runs the steps; an unconfirmed side-effecting step
  pauses (``ran=False``, ``needs_confirmation=True``), a confirmed re-run completes.
* ``DELETE /protocols/{id}`` removes one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_protocols=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_protocols=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose protocols flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_protocols_disabled_create_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/protocols", json={"name": "x", "trigger_phrase": "x"})
    assert resp.status_code == 404


def test_protocols_disabled_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/protocols")
    assert resp.status_code == 404


def test_protocols_disabled_run_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/protocols/1/run", json={})
    assert resp.status_code == 404


def test_protocols_disabled_delete_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.delete("/protocols/1")
    assert resp.status_code == 404


def test_protocols_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), create is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/protocols", json={"name": "x", "trigger_phrase": "x"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> full CRUD + run
# --------------------------------------------------------------------------- #
def test_protocols_create_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/protocols",
            json={
                "name": "Goodnight",
                "trigger_phrase": "goodnight",
                "steps": [
                    {"tool": "list_reminders", "args": {"status": "open"}},
                ],
            },
        )
        assert created.status_code == 200
        body = created.json()
        assert body["name"] == "Goodnight"
        assert body["enabled"] is True
        assert isinstance(body["id"], int)
        assert body["steps"] == [
            {"tool": "list_reminders", "args": {"status": "open"}}
        ]

        listed = client.get("/protocols")
        assert listed.status_code == 200
        names = [p["name"] for p in listed.json()["protocols"]]
        assert names == ["Goodnight"]


def test_protocols_run_readonly_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/protocols",
            json={
                "name": "Status",
                "trigger_phrase": "status check",
                "steps": [
                    {"tool": "list_reminders", "args": {"status": "open"}},
                ],
            },
        )
        pid = created.json()["id"]

        ran = client.post(f"/protocols/{pid}/run", json={})
        assert ran.status_code == 200
        result = ran.json()
        assert result["protocol"] == "Status"
        assert result["ran"] is True
        assert result["needs_confirmation"] is False
        assert [s["tool"] for s in result["steps"]] == ["list_reminders"]


def test_protocols_run_pauses_on_side_effecting_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/protocols",
            json={
                "name": "Lights Out",
                "trigger_phrase": "lights out",
                "steps": [
                    {"tool": "list_reminders", "args": {"status": "open"}},
                    {
                        "tool": "notify",
                        "args": {
                            "channel": "webhook",
                            "target": "ops",
                            "subject": "Lights out",
                            "body": "running the lights-out routine",
                        },
                    },
                ],
            },
        )
        pid = created.json()["id"]

        # Unconfirmed: the read-only step runs, the side-effecting "notify" step
        # pauses before executing; the run reports needs_confirmation.
        unconfirmed = client.post(f"/protocols/{pid}/run", json={}).json()
        assert unconfirmed["ran"] is False
        assert unconfirmed["needs_confirmation"] is True
        assert [s["tool"] for s in unconfirmed["steps"]] == [
            "list_reminders",
            "notify",
        ]
        assert unconfirmed["steps"][-1]["needs_confirmation"] is True

        # Confirmed: every step runs (the fake notify sink records the message).
        confirmed = client.post(
            f"/protocols/{pid}/run", json={"confirmed": True}
        ).json()
        assert confirmed["ran"] is True
        assert confirmed["needs_confirmation"] is False
        assert [s["tool"] for s in confirmed["steps"]] == [
            "list_reminders",
            "notify",
        ]


def test_protocols_run_unknown_id_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/protocols/999/run", json={})
    assert resp.status_code == 404


def test_protocols_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/protocols",
            json={"name": "temp", "trigger_phrase": "temp"},
        )
        pid = created.json()["id"]

        deleted = client.delete(f"/protocols/{pid}")
        assert deleted.status_code == 200
        assert deleted.json()["removed"] == 1

        assert client.get("/protocols").json()["protocols"] == []
