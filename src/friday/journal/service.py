"""Auto-journaling: a deterministic per-day digest of FRIDAY's activity (Tier 2).

The :class:`JournalService` aggregates a single calendar day's events into one
:class:`JournalEntry` — a structured, deterministic summary assembled entirely
from the *local* stores the rest of FRIDAY already maintains (the process-wide
:class:`~friday.observability.audit.AuditLog` and
:class:`~friday.observability.metrics.Metrics`, and — when it exposes them — the
shared :class:`~friday.reminders.store.SQLiteReminderStore`). A journal entry is
therefore pure assembly with no new infrastructure, mirroring
:class:`~friday.briefing.service.BriefingService`.

Design rules (binding):

* **Clock injected.** ``build_entry(day)`` is driven entirely by the passed
  ``day`` datetime — the entry's ``date`` and every "today" comparison derive
  from it, never the wall clock. Tested paths advance ``day`` deterministically.
* **Deterministic highlights.** The audit rows, completed reminders, and a
  one-line metrics summary are folded into ``highlights`` deterministically (no
  randomness, no clock reads), so the same seeded stores always yield the same
  entry.
* **Graceful degradation.** With the metrics absent that line is simply omitted;
  with a reminder store that does not expose the day's completed reminders the
  reminder line is skipped rather than failing.
* **LLM optional and NON-FATAL.** When an ``llm`` is provided the service may
  narrate a short natural-language ``summary``, but the call is wrapped so *any*
  error (provider failure, timeout, blank text) falls back to the deterministic
  structured summary — building an entry must never raise because the LLM failed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.journal.service")

#: How many tool-call audit rows the highlights surface (one line each).
_AUDIT_HIGHLIGHT_LIMIT = 5


class JournalEntry(BaseModel):
    """One day's journal: a date, a summary, deterministic highlights, a count.

    Attributes:
        date: The journaled calendar day as a ``YYYY-MM-DD`` string.
        summary: A short prose summary — the LLM's narration when one is
            available, otherwise a deterministic structured sentence.
        highlights: Deterministically-assembled bullet lines (tool activity,
            reminders completed, a metrics glance).
        event_count: The number of tool-call audit rows aggregated for the day.
    """

    date: str
    summary: str
    highlights: list[str] = Field(default_factory=list)
    event_count: int = 0


class JournalService:
    """Aggregate a day's local activity into a :class:`JournalEntry`.

    Args:
        audit_log: The process-wide audit log; its recent rows drive the
            tool-activity count + highlight lines.
        reminder_store: Optional shared reminder store. When it exposes the
            reminders completed on the day (a ``list_reminders``/``due`` surface),
            those become a highlight line; otherwise the reminder line is skipped
            gracefully.
        metrics: Optional process-wide metrics; when present its snapshot becomes
            a one-line glance highlight.
        llm: Optional LLM provider; when present a short natural-language summary
            may be narrated. Any LLM error is swallowed (deterministic
            structured-summary fallback) so building an entry never fails.
        owner_address: How the deterministic summary addresses the owner.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        *,
        reminder_store: Any | None = None,
        metrics: Metrics | None = None,
        llm: LLMProvider | None = None,
        owner_address: str = "Boss",
    ) -> None:
        self._audit = audit_log
        self._reminders = reminder_store
        self._metrics = metrics
        self._llm = llm
        self._owner = owner_address

    async def build_entry(self, day: datetime) -> JournalEntry:
        """Build the day's :class:`JournalEntry` (clock injected, never wall-clock).

        Aggregates the day's tool-call audit rows (a count plus a few highlight
        lines), the reminders completed that day (when the reminder store exposes
        them), and a one-line metrics glance into deterministic ``highlights``.
        When an ``llm`` is configured its narration becomes the ``summary``,
        wrapped so any failure falls back to a deterministic structured summary.
        """
        date_str = day.date().isoformat()
        rows = self._audit.recent(limit=self._audit_limit())
        event_count = len(rows)

        highlights: list[str] = []
        highlights.extend(self._tool_highlights(rows))
        reminder_line = self._reminders_highlight(date_str)
        if reminder_line is not None:
            highlights.append(reminder_line)
        metrics_line = self._metrics_highlight()
        if metrics_line is not None:
            highlights.append(metrics_line)

        deterministic = self._structured_summary(date_str, event_count, highlights)
        summary = await self._narrate(date_str, highlights, deterministic)

        return JournalEntry(
            date=date_str,
            summary=summary,
            highlights=highlights,
            event_count=event_count,
        )

    # -- tool-call activity ------------------------------------------------- #
    def _audit_limit(self) -> int:
        """How many recent audit rows the entry inspects (a small fixed window)."""
        return _AUDIT_HIGHLIGHT_LIMIT

    def _tool_highlights(self, rows: list[Any]) -> list[str]:
        """A count line plus one line per recent tool call (deterministic).

        The count line is always present (``"0 tool calls today."`` on an empty
        day); each subsequent line renders a single audit row as ``tool -> ok`` /
        ``tool -> <error_code>``, preserving the audit log's oldest-first order so
        the highlights are stable for the same seeded rows.
        """
        count = len(rows)
        noun = "tool call" if count == 1 else "tool calls"
        lines = [f"{count} {noun} today."]
        for row in rows:
            outcome = "ok" if row.ok else (row.error_code or "error")
            lines.append(f"{row.tool} -> {outcome}")
        return lines

    # -- reminders completed that day --------------------------------------- #
    def _reminders_highlight(self, date_str: str) -> str | None:
        """One line counting the reminders completed on ``date_str``, or ``None``.

        Degrades gracefully: with no reminder store, or a store that does not
        expose a way to enumerate completed reminders, this returns ``None`` so the
        reminder line is simply skipped. A reminder counts as completed on the day
        when it is ``done`` and its ``due_at`` (when present) falls on ``date_str``.
        """
        completed = self._completed_reminders(date_str)
        if completed is None:
            return None
        count = len(completed)
        noun = "reminder" if count == 1 else "reminders"
        return f"{count} {noun} completed today."

    def _completed_reminders(self, date_str: str) -> list[Any] | None:
        """Reminders completed on ``date_str``, or ``None`` if not enumerable.

        Uses the store's ``list_reminders(status="all")`` surface when present
        (the shared :class:`SQLiteReminderStore` exposes it); any other store —
        or one that raises probing it — yields ``None`` so the caller skips the
        line. A reminder counts when it is ``done`` and, if it has a ``due_at``,
        that date is ``date_str`` (an undated done reminder counts as completed
        today since the store does not track a completion timestamp).
        """
        store = self._reminders
        if store is None:
            return None
        lister = getattr(store, "list_reminders", None)
        if not callable(lister):
            return None
        try:
            all_reminders = lister(status="all")
        except Exception:  # noqa: BLE001 - unknown store shape; degrade gracefully
            logger.warning("journal: reminder store probe failed; skipping line")
            return None
        completed: list[Any] = []
        for reminder in all_reminders:
            if getattr(reminder, "status", None) != "done":
                continue
            due_at = getattr(reminder, "due_at", None)
            if due_at is not None and str(due_at)[:10] != date_str:
                continue
            completed.append(reminder)
        return completed

    # -- at-a-glance metrics ------------------------------------------------ #
    def _metrics_highlight(self) -> str | None:
        """One-line metrics glance (requests / tool_calls / errors), or ``None``."""
        if self._metrics is None:
            return None
        snap: dict[str, Any] = self._metrics.snapshot()
        return (
            f"Metrics: {snap.get('requests', 0)} requests, "
            f"{snap.get('tool_calls', 0)} tool calls, "
            f"{snap.get('errors', 0)} errors."
        )

    # -- summary: deterministic + optional LLM narration -------------------- #
    def _structured_summary(
        self, date_str: str, event_count: int, highlights: list[str]
    ) -> str:
        """A deterministic one-line summary — the never-fail fallback.

        Derived purely from the date, the event count, and the highlight count, so
        it is stable for the same inputs and addresses the owner.
        """
        noun = "event" if event_count == 1 else "events"
        return (
            f"Journal for {date_str}, {self._owner}: {event_count} {noun} "
            f"recorded across {len(highlights)} highlight(s)."
        )

    async def _narrate(
        self, date_str: str, highlights: list[str], deterministic: str
    ) -> str:
        """Optionally narrate a short summary; never raise.

        When no ``llm`` is configured this returns the deterministic summary
        unchanged. When an ``llm`` is present its completion is wrapped in a broad
        ``except`` so any failure — provider error, timeout, blank text — degrades
        to the deterministic structured summary rather than failing the entry.
        """
        if self._llm is None:
            return deterministic
        prompt = self._summary_prompt(date_str, highlights)
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
            text = (response.text or "").strip()
        except Exception:  # noqa: BLE001 - LLM is optional + non-fatal
            logger.warning("journal LLM narration failed; using structured summary")
            return deterministic
        return text or deterministic

    @staticmethod
    def _summary_prompt(date_str: str, highlights: list[str]) -> str:
        """Render the day's highlights into a compact summarization prompt."""
        body = "\n".join(f"- {line}" for line in highlights)
        return (
            f"Summarize FRIDAY's activity log for {date_str} in two or three "
            "friendly sentences. Be concise and do not invent details.\n\n" + body
        )
