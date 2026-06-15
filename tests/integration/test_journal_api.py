"""Integration tests for the ``/journal`` API + the scheduler journal action.

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_JOURNAL`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the briefing / reminders / scheduler API tests). No
network, no key.

Covered:
* ``GET /journal`` is ``404`` when the flag is off (default off too).
* ``POST /journal/build`` -> ``GET /journal/{date}`` -> ``GET /journal`` round-trip
  when enabled.
* The scheduler ``"journal"`` action builds + saves an entry into the shared
  journal store (run via the scheduler's run-now path over a built runtime,
  clock-injected through ``build_entry``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from friday.app import build_runtime, create_app
from friday.config import Settings
from friday.scheduler.store import Trigger


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_journal=True,
        # Keep reminders + scheduler on so the shared stores/actions are present.
        enable_reminders=True,
        enable_scheduler=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_journal=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose journal flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404
# --------------------------------------------------------------------------- #
def test_journal_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        assert client.get("/journal").status_code == 404
        assert client.get("/journal/2026-06-15").status_code == 404
        assert client.post("/journal/build", json={}).status_code == 404


def test_journal_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), GET /journal is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        assert client.get("/journal").status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> build + get + list round-trip
# --------------------------------------------------------------------------- #
def test_journal_build_then_get_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        # Build the entry for an explicit date.
        build = client.post("/journal/build", json={"date": "2026-06-15"})
        assert build.status_code == 200
        entry = build.json()
        assert entry["date"] == "2026-06-15"
        assert "summary" in entry and entry["summary"]
        assert isinstance(entry["highlights"], list)
        assert "event_count" in entry

        # GET /journal/{date} returns the saved entry.
        got = client.get("/journal/2026-06-15")
        assert got.status_code == 200
        assert got.json()["date"] == "2026-06-15"

        # GET /journal lists it.
        listing = client.get("/journal")
        assert listing.status_code == 200
        body = listing.json()
        assert body["count"] == 1
        assert body["entries"][0]["date"] == "2026-06-15"


def test_journal_build_defaults_to_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """An omitted date defaults to UTC today and saves under it."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        build = client.post("/journal/build", json={})
        assert build.status_code == 200
        date = build.json()["date"]
        assert client.get(f"/journal/{date}").status_code == 200


def test_journal_build_upserts_by_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-building the same date overwrites rather than duplicating."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        client.post("/journal/build", json={"date": "2026-06-15"})
        client.post("/journal/build", json={"date": "2026-06-15"})
        assert client.get("/journal").json()["count"] == 1


def test_journal_get_missing_date_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        assert client.get("/journal/2099-01-01").status_code == 404


def test_journal_build_bad_date_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/journal/build", json={"date": "06/15/2026"})
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Scheduler "journal" action builds + saves into the shared store
# --------------------------------------------------------------------------- #
async def test_scheduler_journal_action_builds_and_saves() -> None:
    runtime = build_runtime(_enable_settings())
    # Seed an audit row so the built entry has aggregated activity.
    runtime.audit.record(
        correlation_id="c1", tool="web_search", args={}, ok=True, error_code=None
    )

    assert runtime.journal_store.list_entries() == []

    trigger = Trigger(
        id=1,
        name="eod",
        kind="daily",
        spec="22:00",
        action="journal",
        enabled=True,
    )
    ran = await runtime.scheduler.run_action(trigger)
    assert ran is True

    # Exactly one entry landed in the shared store for today's UTC date.
    entries = runtime.journal_store.list_entries()
    assert len(entries) == 1
    today = datetime.now(UTC).date().isoformat()
    assert entries[0].date == today
    assert entries[0].event_count == 1


async def test_scheduler_journal_action_uses_shared_journal_service() -> None:
    runtime = build_runtime(_enable_settings())
    # The route service and the scheduler action share one journal service +
    # store, so a built entry is readable straight back from the store.
    entry = await runtime.journal_service.build_entry(
        datetime(2026, 6, 15, 18, 0, tzinfo=UTC)
    )
    runtime.journal_store.save(entry)
    fetched = runtime.journal_store.get("2026-06-15")
    assert fetched is not None
    assert fetched.date == "2026-06-15"
