"""Auto-journaling (Tier 2): a deterministic per-day digest of FRIDAY's activity.

This package owns FRIDAY's journaling feature — a structured daily entry
(``date`` + ``summary`` + deterministic ``highlights`` + ``event_count``)
aggregated from the local stores the rest of FRIDAY already maintains (the
process-wide audit log + metrics, and the shared reminder store when it exposes
the day's completed reminders). Entries are buildable on demand (the flagged
``POST /journal/build``), readable back (``GET /journal`` / ``GET
/journal/{date}``), and producible by a scheduler ``"journal"`` action so an
end-of-day entry writes itself. Off by default behind ``FRIDAY_ENABLE_JOURNAL``.
Optional LLM narration is non-fatal: any LLM error falls back to a deterministic
structured summary.

The public surface is the typed :class:`~friday.journal.service.JournalEntry`
model, the :class:`~friday.journal.service.JournalService` aggregator, and the
durable :class:`~friday.journal.store.SQLiteJournalStore`.
"""

from __future__ import annotations

from friday.journal.service import JournalEntry, JournalService
from friday.journal.store import SQLiteJournalStore

__all__ = ["JournalEntry", "JournalService", "SQLiteJournalStore"]
