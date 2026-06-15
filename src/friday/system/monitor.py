"""Hardware/system monitor: a snapshot sampler + threshold checker (Tier 2).

Three pieces, all offline-testable via dependency injection:

* :class:`SystemStats` — a pydantic v2 snapshot of the host's CPU / memory / disk
  utilisation, plus the optional temperature and 1-minute load average and the
  UTC timestamp the snapshot was taken.
* :class:`Sampler` — a structural protocol: ``def sample() -> SystemStats``. The
  real :class:`PsutilSampler` reads live values (lazy-importing ``psutil`` so the
  module imports even where the dependency is absent, and guarding every optional
  metric with ``try/except -> None``); tests inject a *fake* sampler returning
  controlled snapshots, so they never touch ``psutil`` and stay deterministic.
* :class:`SystemMonitor` — wraps a sampler with four thresholds; ``stats()``
  returns the current snapshot unchanged, and ``check()`` returns one
  :class:`Alert` per breached threshold (``[]`` when healthy). The breach test is
  strict ``>`` (a value *equal* to its threshold is healthy), and an optional
  metric that is ``None`` (sensor unavailable) never breaches.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SystemStats(BaseModel):
    """A point-in-time snapshot of host resource utilisation.

    ``cpu_percent`` / ``mem_percent`` / ``disk_percent`` are 0..100 utilisation
    percentages. ``temp_c`` (CPU package temperature in °C) and ``load1`` (the
    1-minute load average) are optional — ``None`` when the platform/sensor does
    not expose them — so a breach check skips them rather than fabricating. ``ts``
    is the ISO-8601 UTC instant the snapshot was sampled.
    """

    cpu_percent: float
    mem_percent: float
    disk_percent: float
    temp_c: float | None = None
    load1: float | None = None
    ts: str


class Alert(BaseModel):
    """One breached threshold: the metric, its sampled value, and the limit.

    ``metric`` is the :class:`SystemStats` field name that breached
    (``"cpu_percent"`` / ``"mem_percent"`` / ``"disk_percent"`` / ``"temp_c"``);
    ``value`` is the sampled reading, ``threshold`` the configured limit it
    exceeded, and ``message`` a ready-to-emit human-readable line.
    """

    metric: str
    value: float
    threshold: float
    message: str


@runtime_checkable
class Sampler(Protocol):
    """Anything that can produce a :class:`SystemStats` snapshot on demand.

    The single seam the monitor reads through: the real :class:`PsutilSampler`
    in production, a controlled fake in tests. Structural (``Protocol``) so a
    test fake needs only a ``sample()`` method, no inheritance.
    """

    def sample(self) -> SystemStats:
        """Return a fresh snapshot of current host resource utilisation."""
        ...


class PsutilSampler:
    """The real sampler: live readings via ``psutil`` (lazy-imported).

    ``psutil`` is imported *inside* :meth:`sample`, never at module import, so
    this module (and therefore the whole app) imports even where ``psutil`` is
    not installed — the dependency is only required when a live sample is taken.
    The always-present metrics (CPU / memory / disk) come from
    ``cpu_percent`` / ``virtual_memory`` / ``disk_usage``; the optional ones
    (temperature via ``sensors_temperatures``, load via ``getloadavg``) are each
    guarded with ``try/except -> None`` because they are platform-specific.

    ``disk_path`` (default ``"/"``) is the mount whose usage is reported, and
    ``now`` is an injectable clock (default :func:`datetime.now` in UTC) so the
    snapshot timestamp can be pinned in a test without touching ``psutil``.
    """

    def __init__(self, *, disk_path: str = "/") -> None:
        self._disk_path = disk_path

    def sample(self) -> SystemStats:
        """Read live host metrics; optional sensors degrade to ``None``."""
        import psutil  # lazy: keep the module import-safe without the dependency

        cpu_percent = float(psutil.cpu_percent(interval=None))
        mem_percent = float(psutil.virtual_memory().percent)
        disk_percent = float(psutil.disk_usage(self._disk_path).percent)
        return SystemStats(
            cpu_percent=cpu_percent,
            mem_percent=mem_percent,
            disk_percent=disk_percent,
            temp_c=_read_temp(psutil),
            load1=_read_load1(psutil),
            ts=datetime.now(UTC).isoformat(),
        )


def _read_temp(psutil_mod: object) -> float | None:
    """Best-effort CPU package temperature in °C, or ``None`` if unavailable.

    ``sensors_temperatures`` is Linux-only and may be absent or empty; any
    failure (missing attribute, empty mapping, unexpected shape) degrades to
    ``None`` so a platform without sensors simply reports no temperature.
    """
    try:
        readings = psutil_mod.sensors_temperatures()  # type: ignore[attr-defined]
    except (AttributeError, OSError, NotImplementedError):
        return None
    if not readings:
        return None
    # Prefer a CPU-ish sensor group, else the first group with a reading.
    for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
        entries = readings.get(key)
        if entries:
            return float(entries[0].current)
    for entries in readings.values():
        if entries:
            return float(entries[0].current)
    return None


def _read_load1(psutil_mod: object) -> float | None:
    """The 1-minute load average, or ``None`` where ``getloadavg`` is unavailable.

    ``getloadavg`` is not implemented on every platform (notably some Windows
    builds); any failure degrades to ``None``.
    """
    try:
        load1, _load5, _load15 = psutil_mod.getloadavg()  # type: ignore[attr-defined]
    except (AttributeError, OSError, NotImplementedError):
        return None
    return float(load1)


class SystemMonitor:
    """A sampler plus four thresholds: ``stats()`` snapshots, ``check()`` alerts.

    The thresholds default to the documented limits (CPU/mem 90%, disk 95%,
    temperature 85 °C); each is overridable per instance (the app wires them from
    settings). A breach is strict ``>`` — a value *equal* to its threshold is
    healthy — and an optional metric (``temp_c``) that is ``None`` never breaches.
    ``check()`` reports load average for context only; there is no load threshold,
    so ``load1`` is never an alert source.
    """

    def __init__(
        self,
        sampler: Sampler,
        *,
        cpu_threshold: float = 90.0,
        mem_threshold: float = 90.0,
        disk_threshold: float = 95.0,
        temp_threshold: float = 85.0,
    ) -> None:
        self._sampler = sampler
        self._cpu_threshold = cpu_threshold
        self._mem_threshold = mem_threshold
        self._disk_threshold = disk_threshold
        self._temp_threshold = temp_threshold

    def stats(self) -> SystemStats:
        """Return the current snapshot, unmodified, from the injected sampler."""
        return self._sampler.sample()

    def check(self) -> list[Alert]:
        """Return one :class:`Alert` per breached threshold (``[]`` when healthy).

        Samples once, then tests CPU / memory / disk / temperature against their
        thresholds with a strict ``>`` (boundary-equal is healthy). The optional
        temperature is skipped when ``None``. Order is stable: cpu, mem, disk,
        temp.
        """
        stats = self._sampler.sample()
        alerts: list[Alert] = []
        alerts.extend(
            _breach("cpu_percent", stats.cpu_percent, self._cpu_threshold, "CPU")
        )
        alerts.extend(
            _breach("mem_percent", stats.mem_percent, self._mem_threshold, "Memory")
        )
        alerts.extend(
            _breach("disk_percent", stats.disk_percent, self._disk_threshold, "Disk")
        )
        if stats.temp_c is not None:
            alerts.extend(
                _breach("temp_c", stats.temp_c, self._temp_threshold, "Temperature")
            )
        return alerts


def _breach(
    metric: str, value: float, threshold: float, label: str
) -> list[Alert]:
    """A single-element alert list when ``value > threshold``, else empty.

    Returns a list (not an :class:`Alert` ``| None``) so callers can ``extend``
    uniformly. The breach test is strict ``>`` so a reading equal to the
    threshold is not flagged.
    """
    if value > threshold:
        return [
            Alert(
                metric=metric,
                value=value,
                threshold=threshold,
                message=f"{label} at {value:g} exceeds threshold {threshold:g}",
            )
        ]
    return []
