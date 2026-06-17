"""Unit tests for the hardware/system monitor (Tier 2 system module).

Fully offline and DETERMINISTIC: every test injects a *fake* sampler that returns
controlled :class:`~friday.system.monitor.SystemStats`, so no real ``psutil``
read ever happens here and the assertions never depend on the host machine's
actual CPU/memory/disk/temperature. The real :class:`PsutilSampler` is exercised
only for its import-safety (it must import even if ``psutil`` were absent, since
the import is lazy inside ``sample``), never for live values.

Pinned behaviours:

* ``stats()`` returns exactly what the injected sampler produced.
* ``check()`` returns one :class:`~friday.system.monitor.Alert` per breached
  threshold and ``[]`` when every metric is healthy.
* Threshold boundary: a value *equal* to the threshold is NOT a breach; a value
  strictly *greater* than the threshold IS.
* Optional metrics (``temp_c`` / ``load1``) are ``None``-safe: a ``None`` never
  breaches its threshold.
"""

from __future__ import annotations

import pytest

from friday.system.monitor import (
    Alert,
    PsutilSampler,
    Sampler,
    SystemMonitor,
    SystemStats,
)


def _stats(
    *,
    cpu: float = 10.0,
    mem: float = 20.0,
    disk: float = 30.0,
    temp: float | None = 40.0,
    load1: float | None = 0.5,
    ts: str = "2026-06-15T12:00:00+00:00",
) -> SystemStats:
    """A controlled, healthy-by-default stats snapshot for the fake sampler."""
    return SystemStats(
        cpu_percent=cpu,
        mem_percent=mem,
        disk_percent=disk,
        temp_c=temp,
        load1=load1,
        ts=ts,
    )


class _FakeSampler:
    """A deterministic :class:`Sampler` returning a fixed snapshot (no psutil)."""

    def __init__(self, stats: SystemStats) -> None:
        self._stats = stats
        self.calls = 0

    def sample(self) -> SystemStats:
        self.calls += 1
        return self._stats


# --------------------------------------------------------------------------- #
# protocol / typing
# --------------------------------------------------------------------------- #
def test_psutil_sampler_uses_blocking_cpu_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    # cpu_percent(interval=None) returns 0.0 on the first call and is otherwise
    # cadence-dependent; the sampler must pass a small blocking interval so the
    # reading reflects real instantaneous load.
    import sys
    import types

    calls: list[float | None] = []
    fake = types.SimpleNamespace(
        cpu_percent=lambda interval=None: (calls.append(interval) or 0.0),
        virtual_memory=lambda: types.SimpleNamespace(percent=10.0),
        disk_usage=lambda path: types.SimpleNamespace(percent=10.0),
        sensors_temperatures=lambda: {},
        getloadavg=lambda: (0.0, 0.0, 0.0),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)

    PsutilSampler().sample()

    assert calls == [0.1]


def test_fake_sampler_satisfies_protocol() -> None:
    """The fake (and the real PsutilSampler) structurally satisfy ``Sampler``."""
    fake: Sampler = _FakeSampler(_stats())
    assert isinstance(fake.sample(), SystemStats)
    real: Sampler = PsutilSampler()
    assert isinstance(real, Sampler)


# --------------------------------------------------------------------------- #
# stats() — passthrough of the injected sampler
# --------------------------------------------------------------------------- #
def test_stats_returns_injected_sampler_snapshot() -> None:
    snapshot = _stats(cpu=12.5, mem=34.0, disk=56.0, temp=41.0, load1=1.25)
    sampler = _FakeSampler(snapshot)
    monitor = SystemMonitor(sampler)

    result = monitor.stats()

    assert result == snapshot
    assert result.cpu_percent == 12.5
    assert result.mem_percent == 34.0
    assert result.disk_percent == 56.0
    assert result.temp_c == 41.0
    assert result.load1 == 1.25
    assert result.ts == "2026-06-15T12:00:00+00:00"
    assert sampler.calls == 1


# --------------------------------------------------------------------------- #
# check() — one Alert per breached threshold; [] when healthy
# --------------------------------------------------------------------------- #
def test_check_returns_empty_when_all_healthy() -> None:
    monitor = SystemMonitor(_FakeSampler(_stats()))
    assert monitor.check() == []


def test_check_returns_one_alert_per_breached_threshold() -> None:
    # Every metric strictly over its (custom) threshold -> four alerts.
    breached = _stats(cpu=99.0, mem=98.0, disk=97.0, temp=96.0)
    monitor = SystemMonitor(
        _FakeSampler(breached),
        cpu_threshold=90.0,
        mem_threshold=90.0,
        disk_threshold=95.0,
        temp_threshold=85.0,
    )

    alerts = monitor.check()

    by_metric = {alert.metric: alert for alert in alerts}
    assert set(by_metric) == {"cpu_percent", "mem_percent", "disk_percent", "temp_c"}
    for alert in alerts:
        assert isinstance(alert, Alert)
        assert alert.value > alert.threshold
        assert alert.message  # a human-readable line is always present
    assert by_metric["cpu_percent"].value == 99.0
    assert by_metric["cpu_percent"].threshold == 90.0
    assert by_metric["disk_percent"].threshold == 95.0
    assert by_metric["temp_c"].threshold == 85.0


def test_check_alerts_only_breached_metrics() -> None:
    # Only CPU is over; the rest are healthy -> exactly one alert.
    stats = _stats(cpu=95.0, mem=10.0, disk=10.0, temp=10.0)
    monitor = SystemMonitor(_FakeSampler(stats), cpu_threshold=90.0)

    alerts = monitor.check()

    assert [alert.metric for alert in alerts] == ["cpu_percent"]
    assert alerts[0].value == 95.0
    assert alerts[0].threshold == 90.0


# --------------------------------------------------------------------------- #
# threshold boundary — == is NOT a breach, > IS
# --------------------------------------------------------------------------- #
def test_value_equal_to_threshold_is_not_a_breach() -> None:
    stats = _stats(cpu=90.0, mem=90.0, disk=95.0, temp=85.0)
    monitor = SystemMonitor(
        _FakeSampler(stats),
        cpu_threshold=90.0,
        mem_threshold=90.0,
        disk_threshold=95.0,
        temp_threshold=85.0,
    )
    assert monitor.check() == []


def test_value_just_above_threshold_is_a_breach() -> None:
    stats = _stats(cpu=90.1)
    monitor = SystemMonitor(_FakeSampler(stats), cpu_threshold=90.0)
    alerts = monitor.check()
    assert [alert.metric for alert in alerts] == ["cpu_percent"]


# --------------------------------------------------------------------------- #
# optional metrics — None never breaches
# --------------------------------------------------------------------------- #
def test_none_temperature_never_breaches() -> None:
    # temp_c is None (sensor unavailable) -> no temp alert even with a low threshold.
    stats = _stats(temp=None)
    monitor = SystemMonitor(_FakeSampler(stats), temp_threshold=0.0)
    assert [alert.metric for alert in monitor.check()] == []


def test_default_thresholds_match_contract() -> None:
    # Just under the documented defaults (cpu/mem 90, disk 95, temp 85) -> healthy.
    stats = _stats(cpu=89.9, mem=89.9, disk=94.9, temp=84.9)
    monitor = SystemMonitor(_FakeSampler(stats))
    assert monitor.check() == []
