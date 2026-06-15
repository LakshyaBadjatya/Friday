"""Integration tests for the ``/reminders`` REST API (Tier 1).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_REMINDERS`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the RAG / studio API tests). No network, no key.

Covered:
* Every ``/reminders`` surface is ``404`` when the flag is off.
* ``POST /reminders`` creates and returns the reminder.
* ``GET /reminders?status=open|all`` lists, soonest-due first, and the ``all``
  filter includes completed reminders.
* ``POST /reminders/{id}/complete`` completes a reminder.
* ``DELETE /reminders/{id}`` removes one.
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
        enable_reminders=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_reminders=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose reminders flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_reminders_disabled_create_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/reminders", json={"text": "x"})
    assert resp.status_code == 404


def test_reminders_disabled_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/reminders")
    assert resp.status_code == 404


def test_reminders_disabled_complete_is_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/reminders/1/complete")
    assert resp.status_code == 404


def test_reminders_disabled_delete_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.delete("/reminders/1")
    assert resp.status_code == 404


def test_reminders_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), create is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/reminders", json={"text": "x"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> full CRUD
# --------------------------------------------------------------------------- #
def test_reminders_create_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/reminders",
            json={"text": "call the dentist", "due_at": "2026-06-16T09:00:00+00:00"},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["text"] == "call the dentist"
        assert body["status"] == "open"
        assert isinstance(body["id"], int)

        listed = client.get("/reminders?status=open")
        assert listed.status_code == 200
        items = listed.json()["reminders"]
        assert [r["text"] for r in items] == ["call the dentist"]


def test_reminders_list_orders_soonest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        client.post(
            "/reminders",
            json={"text": "later", "due_at": "2026-06-20T00:00:00+00:00"},
        )
        client.post(
            "/reminders",
            json={"text": "sooner", "due_at": "2026-06-16T00:00:00+00:00"},
        )
        listed = client.get("/reminders")
        texts = [r["text"] for r in listed.json()["reminders"]]
        assert texts == ["sooner", "later"]


def test_reminders_complete_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/reminders",
            json={"text": "finish me", "due_at": "2026-06-16T00:00:00+00:00"},
        )
        rid = created.json()["id"]

        done = client.post(f"/reminders/{rid}/complete")
        assert done.status_code == 200
        assert done.json()["completed"] is True

        # Now absent from the open list, present in ``all``.
        assert client.get("/reminders?status=open").json()["reminders"] == []
        all_items = client.get("/reminders?status=all").json()["reminders"]
        assert [r["text"] for r in all_items] == ["finish me"]
        assert all_items[0]["status"] == "done"


def test_reminders_complete_unknown_id_is_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/reminders/999/complete")
    assert resp.status_code == 404


def test_reminders_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post("/reminders", json={"text": "temp"})
        rid = created.json()["id"]

        deleted = client.delete(f"/reminders/{rid}")
        assert deleted.status_code == 200
        assert deleted.json()["removed"] == 1

        assert client.get("/reminders?status=all").json()["reminders"] == []
