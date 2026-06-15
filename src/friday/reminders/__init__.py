"""Reminders & tasks (Tier 1): a local-first, SQLite-backed reminder store.

This package owns FRIDAY's reminder feature — durable reminders with optional
due dates and simple recurrence — usable directly (the flagged ``/reminders``
REST surface) and by the Automation agent (via the tool registry). It reuses the
Phase-4 SQLite path (``memory_db_path``) and is off by default behind
``FRIDAY_ENABLE_REMINDERS``.

The public surface is the typed :class:`~friday.reminders.store.Reminder` model
and the :class:`~friday.reminders.store.SQLiteReminderStore` adapter.
"""

from __future__ import annotations

from friday.reminders.store import Reminder, ReminderStore, SQLiteReminderStore

__all__ = ["Reminder", "ReminderStore", "SQLiteReminderStore"]
