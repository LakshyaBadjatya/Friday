"""Integration tests for the ``/briefing`` API + the scheduler briefing action.

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_BRIEFING`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the reminders / scheduler / RAG / studio API tests).
No network, no key.

Covered:
* ``GET /briefing`` is ``404`` when the flag is off (default off too).
* ``GET /briefing`` enabled returns a structured briefing (greeting + sections).
* The scheduler ``"briefing"`` action builds a briefing and emits it to the
  shared notify sink (run via the scheduler's run-now path over a built runtime,
  clock-injected through ``build``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from friday.app import build_runtime, create_app
from friday.config import Settings
from friday.scheduler.store import Trigger
from friday.tools.notify import NotifyTool


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_briefing=True,
        # Keep reminders + scheduler on so the shared stores/actions are present.
        enable_reminders=True,
        enable_scheduler=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_briefing=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose briefing flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404
# --------------------------------------------------------------------------- #
def test_briefing_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/briefing")
    assert resp.status_code == 404


def test_briefing_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), GET /briefing is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/briefing")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> returns a structured briefing
# --------------------------------------------------------------------------- #
def test_briefing_enabled_returns_briefing(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        # Seed a reminder so the briefing has content to bucket.
        client.post(
            "/reminders",
            json={"text": "call the dentist", "due_at": "2000-01-01T00:00:00+00:00"},
        )
        resp = client.get("/briefing")

    assert resp.status_code == 200
    body = resp.json()
    assert "generated_at" in body
    assert isinstance(body["greeting"], str) and body["greeting"]
    titles = [section["title"] for section in body["sections"]]
    # The three reminder buckets are always present.
    assert "Overdue" in titles
    assert "Due today" in titles
    assert "Upcoming" in titles
    # The seeded (long-overdue) reminder surfaced in the briefing somewhere.
    assert any(
        "call the dentist" in item
        for section in body["sections"]
        for item in section["items"]
    )


# --------------------------------------------------------------------------- #
# Scheduler "briefing" action builds + emits to the notify sink
# --------------------------------------------------------------------------- #
async def test_scheduler_briefing_action_emits_to_notify_sink() -> None:
    runtime = build_runtime(_enable_settings())

    # The shared notify tool's sink is where the action emits.
    notify = runtime.registry.get("notify")
    assert isinstance(notify, NotifyTool)
    assert notify.sink == []

    # Fire the registered "briefing" action through the scheduler's run-now path.
    trigger = Trigger(
        id=1,
        name="morning",
        kind="daily",
        spec="07:00",
        action="briefing",
        enabled=True,
    )
    ran = await runtime.scheduler.run_action(trigger)
    assert ran is True

    # Exactly one briefing message landed in the shared sink.
    assert len(notify.sink) == 1
    message = notify.sink[0]
    assert message.subject == "Briefing"
    assert message.target == "scheduler"
    # The body carries the greeting + the reminder section headings.
    assert "Overdue" in message.body
    assert "Due today" in message.body
    assert "Upcoming" in message.body


async def test_scheduler_briefing_action_uses_shared_reminder_store() -> None:
    runtime = build_runtime(_enable_settings())
    # Seed an overdue reminder directly into the shared store the action reads.
    runtime.reminder_store.add(
        "pay rent", due_at="2000-01-01T00:00:00+00:00"
    )

    trigger = Trigger(
        id=2,
        name="eod",
        kind="daily",
        spec="18:00",
        action="briefing",
        enabled=True,
    )
    ran = await runtime.scheduler.run_action(trigger)
    assert ran is True

    notify = runtime.registry.get("notify")
    assert isinstance(notify, NotifyTool)
    assert len(notify.sink) == 1
    assert "pay rent" in notify.sink[0].body


def test_briefing_build_uses_injected_now_not_wallclock() -> None:
    """The build is clock-injected: ``generated_at`` echoes the passed ``now``."""
    import asyncio

    runtime = build_runtime(_enable_settings())
    now = datetime(2026, 6, 15, 6, 30, tzinfo=UTC)
    briefing = asyncio.run(runtime.briefing.build(now))
    assert briefing.generated_at == now.isoformat()
    assert "morning" in briefing.greeting.lower()
