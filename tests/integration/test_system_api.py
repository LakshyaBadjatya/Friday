"""Integration tests for the ``/system`` API + the scheduler system_check action.

Offline and DETERMINISTIC: no real ``psutil`` read ever happens. The app builds a
real :class:`~friday.system.monitor.SystemMonitor` over a
:class:`~friday.system.monitor.PsutilSampler`, but every enabled test *replaces*
the monitor's sampler with a controlled fake (so the snapshot is fixed) before
hitting the route or firing the action. The ``TestClient``'s
``FRIDAY_ENABLE_SYSTEM_MONITOR`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the study / briefing / reminders API tests). No
network, no key, no host-dependent assertions.

Covered:
* ``GET /system/stats`` and ``GET /system/check`` are ``404`` when the flag is off
  (default off too).
* ``GET /system/stats`` enabled returns the injected snapshot.
* ``GET /system/check`` enabled returns one alert per breached threshold, ``[]``
  when healthy.
* The scheduler ``"system_check"`` action samples the shared monitor and emits one
  notify-sink message per breached threshold (and nothing when healthy).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import build_runtime, create_app
from friday.config import Settings
from friday.scheduler.store import Trigger
from friday.system.monitor import Sampler, SystemMonitor, SystemStats
from friday.tools.notify import NotifyTool


def _stats(
    *,
    cpu: float = 10.0,
    mem: float = 20.0,
    disk: float = 30.0,
    temp: float | None = 40.0,
    load1: float | None = 0.5,
) -> SystemStats:
    """A controlled snapshot for the injected fake sampler (no psutil)."""
    return SystemStats(
        cpu_percent=cpu,
        mem_percent=mem,
        disk_percent=disk,
        temp_c=temp,
        load1=load1,
        ts="2026-06-15T12:00:00+00:00",
    )


class _FakeSampler:
    """A deterministic :class:`Sampler` returning a fixed snapshot."""

    def __init__(self, stats: SystemStats) -> None:
        self._stats = stats

    def sample(self) -> SystemStats:
        return self._stats


def _inject_sampler(monitor: SystemMonitor, sampler: Sampler) -> None:
    """Swap the monitor's sampler for a fake one (the test-only deterministic seam).

    The app builds the monitor over the real ``PsutilSampler``; replacing its
    sampler keeps the thresholds the app wired from settings while making the
    snapshot fully controlled, so the test never reads the host machine.
    """
    monitor._sampler = sampler  # noqa: SLF001 - intentional test injection seam


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_system_monitor=True,
        # Keep the scheduler on so the wired ``system_check`` action is registered.
        enable_scheduler=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_system_monitor=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose system-monitor flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_system_disabled_stats_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/system/stats")
    assert resp.status_code == 404


def test_system_disabled_check_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/system/check")
    assert resp.status_code == 404


def test_system_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), the routes are 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        stats = client.get("/system/stats")
        check = client.get("/system/check")
    assert stats.status_code == 404
    assert check.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> stats returns the injected snapshot
# --------------------------------------------------------------------------- #
def test_system_stats_returns_injected_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        monitor = client.app.state.system_monitor
        assert isinstance(monitor, SystemMonitor)
        _inject_sampler(monitor, _FakeSampler(_stats(cpu=12.5, mem=34.0, disk=56.0)))

        resp = client.get("/system/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cpu_percent"] == 12.5
    assert body["mem_percent"] == 34.0
    assert body["disk_percent"] == 56.0
    assert body["temp_c"] == 40.0
    assert body["load1"] == 0.5
    assert body["ts"] == "2026-06-15T12:00:00+00:00"


# --------------------------------------------------------------------------- #
# Enabled -> check returns alerts / []
# --------------------------------------------------------------------------- #
def test_system_check_healthy_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        monitor = client.app.state.system_monitor
        _inject_sampler(monitor, _FakeSampler(_stats()))  # all well under defaults

        resp = client.get("/system/check")

    assert resp.status_code == 200
    body = resp.json()
    assert body["alerts"] == []
    assert body["count"] == 0


def test_system_check_reports_breached_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        monitor = client.app.state.system_monitor
        # CPU + memory over the default thresholds (90), disk + temp healthy.
        _inject_sampler(
            monitor, _FakeSampler(_stats(cpu=97.0, mem=95.0, disk=10.0, temp=20.0))
        )

        resp = client.get("/system/check")

    assert resp.status_code == 200
    body = resp.json()
    metrics = {alert["metric"] for alert in body["alerts"]}
    assert metrics == {"cpu_percent", "mem_percent"}
    assert body["count"] == 2
    for alert in body["alerts"]:
        assert alert["value"] > alert["threshold"]
        assert alert["message"]


# --------------------------------------------------------------------------- #
# Scheduler "system_check" action emits breaches to the notify sink
# --------------------------------------------------------------------------- #
async def test_scheduler_system_check_action_emits_breaches() -> None:
    runtime = build_runtime(_enable_settings())
    # Inject a breached snapshot into the SAME monitor the action closes over.
    _inject_sampler(
        runtime.system_monitor, _FakeSampler(_stats(cpu=99.0, disk=99.0))
    )

    notify = runtime.registry.get("notify")
    assert isinstance(notify, NotifyTool)
    assert notify.sink == []

    trigger = Trigger(
        id=1,
        name="resource-watch",
        kind="interval",
        spec="60",
        action="system_check",
        enabled=True,
    )
    ran = await runtime.scheduler.run_action(trigger)
    assert ran is True

    # One message per breached threshold (cpu + disk over their defaults).
    subjects = {msg.subject for msg in notify.sink}
    assert subjects == {"System alert"}
    assert {msg.target for msg in notify.sink} == {"scheduler"}
    assert len(notify.sink) == 2
    bodies = " ".join(msg.body for msg in notify.sink)
    assert "CPU" in bodies
    assert "Disk" in bodies


async def test_scheduler_system_check_action_silent_when_healthy() -> None:
    runtime = build_runtime(_enable_settings())
    _inject_sampler(runtime.system_monitor, _FakeSampler(_stats()))  # healthy

    trigger = Trigger(
        id=2,
        name="resource-watch",
        kind="interval",
        spec="60",
        action="system_check",
        enabled=True,
    )
    ran = await runtime.scheduler.run_action(trigger)
    assert ran is True

    notify = runtime.registry.get("notify")
    assert isinstance(notify, NotifyTool)
    # Healthy host -> nothing emitted.
    assert notify.sink == []
