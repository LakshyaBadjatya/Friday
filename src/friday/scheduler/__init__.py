"""Scheduled triggers — FRIDAY's ambient backbone (Tier 1).

A dependency-free, clock-injectable scheduler that fires time-based triggers
(``interval`` / ``once`` / ``daily`` / ``weekly``) which run named actions, so
reminders go off proactively and briefings can run. SQLite-persisted (survives
restart), off behind ``FRIDAY_ENABLE_SCHEDULER``, and all timing driven by an
injected ``now`` — no wall-clock in tested paths, no cron library.

Public surface:

* :class:`~friday.scheduler.store.Trigger` /
  :class:`~friday.scheduler.store.SQLiteTriggerStore` — the typed row + durable
  store.
* :func:`~friday.scheduler.engine.compute_next_run` — pure next-run math.
* :class:`~friday.scheduler.engine.Scheduler` — action registry + ``tick(now)``.
"""

from __future__ import annotations

from friday.scheduler.engine import Scheduler, compute_next_run
from friday.scheduler.store import SQLiteTriggerStore, Trigger, TriggerKind

__all__ = [
    "Scheduler",
    "SQLiteTriggerStore",
    "Trigger",
    "TriggerKind",
    "compute_next_run",
]
