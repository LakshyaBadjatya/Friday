"""Hardware / system monitoring (Tier 2; default off).

A tiny, fully-injectable monitor over the host's CPU / memory / disk / temperature
/ load. The :class:`~friday.system.monitor.SystemMonitor` reads a snapshot through
an injected :class:`~friday.system.monitor.Sampler` and reports the snapshot
(``stats()``) or the set of breached thresholds (``check()``). The real sampler
(:class:`~friday.system.monitor.PsutilSampler`) lazy-imports ``psutil`` so the
module imports even where ``psutil`` is absent; tests inject a fake sampler and
never touch ``psutil``, keeping them deterministic and offline.
"""

from __future__ import annotations

from friday.system.monitor import (
    Alert,
    PsutilSampler,
    Sampler,
    SystemMonitor,
    SystemStats,
)

__all__ = [
    "Alert",
    "PsutilSampler",
    "Sampler",
    "SystemMonitor",
    "SystemStats",
]
