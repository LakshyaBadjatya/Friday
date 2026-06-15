"""Proactive briefing assembly from local stores (Tier 1).

The :class:`BriefingService` assembles a deterministic digest — due/overdue/
upcoming reminders, recent tool-call activity, and a one-line metrics summary —
plus a time-of-day-aware greeting that addresses the owner. Everything is built
from the *local* stores the rest of FRIDAY already maintains (the shared
:class:`~friday.reminders.store.SQLiteReminderStore`, the process-wide
:class:`~friday.observability.audit.AuditLog` and
:class:`~friday.observability.metrics.Metrics`), so a briefing is pure assembly
with no new infrastructure.

Design rules (binding):

* **Clock injected.** ``build(now)`` is driven entirely by the passed ``now``
  datetime — the reminder buckets, the greeting, and ``generated_at`` all derive
  from it, never the wall clock. Tested paths advance ``now`` deterministically.
* **Deterministic from local stores.** With the audit log / metrics absent the
  briefing degrades gracefully (those sections are simply omitted); the reminder
  sections are always present (possibly empty-worded).
* **LLM optional and NON-FATAL.** When an ``llm`` is provided the service may add
  a short natural-language summary section, but the LLM call is wrapped so *any*
  error (provider failure, timeout, malformed payload) falls back to the
  structured-only briefing — a briefing must never raise because the LLM failed.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.providers.llm import LLMProvider, Message
from friday.reminders.store import Reminder, SQLiteReminderStore

logger = logging.getLogger("friday.briefing.service")

#: How many upcoming (not-yet-due) reminders the upcoming section lists.
_UPCOMING_LIMIT = 5


class BriefingSection(BaseModel):
    """One titled block of the briefing — a heading plus its line items."""

    title: str
    items: list[str] = Field(default_factory=list)


class Briefing(BaseModel):
    """The assembled briefing: when it was built, a greeting, and its sections."""

    generated_at: str
    greeting: str
    sections: list[BriefingSection] = Field(default_factory=list)


class BriefingService:
    """Assemble a :class:`Briefing` from the shared local stores.

    Args:
        reminder_store: The shared reminder store; its :meth:`due` /
            :meth:`list_reminders` drive the overdue / due-today / upcoming
            buckets.
        audit_log: Optional process-wide audit log; when present its recent rows
            become a one-line-each recent-activity section.
        metrics: Optional process-wide metrics; when present its snapshot becomes
            a one-line at-a-glance section.
        llm: Optional LLM provider; when present a short natural-language summary
            section may be added. Any LLM error is swallowed (structured-only
            fallback) so the briefing never fails because of the LLM.
        owner_address: How the greeting addresses the owner (e.g. ``"Boss"``).
        recent_activity_limit: How many recent audit rows to summarize.
    """

    def __init__(
        self,
        reminder_store: SQLiteReminderStore,
        *,
        audit_log: AuditLog | None = None,
        metrics: Metrics | None = None,
        llm: LLMProvider | None = None,
        owner_address: str = "Boss",
        recent_activity_limit: int = 5,
    ) -> None:
        self._reminders = reminder_store
        self._audit = audit_log
        self._metrics = metrics
        self._llm = llm
        self._owner = owner_address
        self._recent_limit = recent_activity_limit

    async def build(self, now: datetime) -> Briefing:
        """Assemble the briefing as of ``now`` (clock injected, never wall-clock).

        Sections, in order: the three reminder buckets (overdue / due today /
        upcoming), an optional recent-activity section (audit log), an optional
        at-a-glance metrics line, and — last — an optional LLM summary. The LLM
        call is wrapped so any failure leaves the structured briefing intact.
        """
        sections: list[BriefingSection] = []
        sections.extend(self._reminder_sections(now))
        recent = self._recent_activity_section()
        if recent is not None:
            sections.append(recent)
        glance = self._metrics_section()
        if glance is not None:
            sections.append(glance)

        briefing = Briefing(
            generated_at=now.isoformat(),
            greeting=self._greeting(now),
            sections=sections,
        )

        summary = await self._llm_summary_section(briefing)
        if summary is not None:
            briefing.sections.append(summary)
        return briefing

    # -- reminder buckets -------------------------------------------------- #
    def _reminder_sections(self, now: datetime) -> list[BriefingSection]:
        """Bucket open reminders into overdue / due today / upcoming by date.

        Bucketing is by ``due_at``'s **calendar date** relative to ``now``'s
        date, independent of the time of day:

        * **Due today** — ``due_at`` falls on ``now``'s date (whether or not the
          time has already passed; a reminder at 17:00 while ``now`` is 08:00 is
          still due today, not upcoming).
        * **Overdue** — ``due_at`` is on a date strictly before today.
        * **Upcoming** — ``due_at`` is on a date strictly after today (capped at
          :data:`_UPCOMING_LIMIT`).

        Undated reminders (``due_at is None``) never appear in any time bucket.
        ``list_reminders(status="open")`` already returns reminders soonest-due
        first, so each bucket preserves that ordering.
        """
        today = now.date().isoformat()

        overdue: list[str] = []
        due_today: list[str] = []
        upcoming: list[str] = []
        for reminder in self._reminders.list_reminders(status="open"):
            if reminder.due_at is None:
                continue
            line = self._reminder_line(reminder)
            due_date = reminder.due_at[:10]
            if due_date == today:
                due_today.append(line)
            elif due_date < today:
                overdue.append(line)
            elif len(upcoming) < _UPCOMING_LIMIT:
                upcoming.append(line)

        return [
            BriefingSection(
                title="Overdue",
                items=overdue or ["Nothing overdue."],
            ),
            BriefingSection(
                title="Due today",
                items=due_today or ["Nothing due today."],
            ),
            BriefingSection(
                title="Upcoming",
                items=upcoming or ["Nothing upcoming."],
            ),
        ]

    @staticmethod
    def _reminder_line(reminder: Reminder) -> str:
        """One reminder rendered as a single line (text + optional due time)."""
        if reminder.due_at is None:
            return reminder.text
        return f"{reminder.text} (due {reminder.due_at})"

    # -- recent activity --------------------------------------------------- #
    def _recent_activity_section(self) -> BriefingSection | None:
        """Summarize the last few audit rows to one line each, or ``None``.

        Returns ``None`` when no audit log was provided so the section is simply
        omitted; an empty audit log yields a single "nothing recorded" line.
        """
        if self._audit is None:
            return None
        rows = self._audit.recent(limit=self._recent_limit)
        if not rows:
            items = ["No recent activity."]
        else:
            items = [
                f"{row.tool} -> {'ok' if row.ok else (row.error_code or 'error')}"
                for row in rows
            ]
        return BriefingSection(title="Recent activity", items=items)

    # -- at-a-glance metrics ----------------------------------------------- #
    def _metrics_section(self) -> BriefingSection | None:
        """One-line metrics summary (requests / tool_calls / errors), or ``None``."""
        if self._metrics is None:
            return None
        snap = self._metrics.snapshot()
        line = (
            f"{snap.get('requests', 0)} requests, "
            f"{snap.get('tool_calls', 0)} tool calls, "
            f"{snap.get('errors', 0)} errors"
        )
        return BriefingSection(title="At a glance", items=[line])

    # -- greeting ---------------------------------------------------------- #
    def _greeting(self, now: datetime) -> str:
        """A time-of-day-aware greeting addressing the owner, derived from ``now``.

        Morning (05:00–11:59), afternoon (12:00–16:59), evening (17:00–20:59),
        and night (21:00–04:59) all yield a distinct phrasing, so the greeting
        varies by ``now``'s hour and always names ``owner_address``.
        """
        hour = now.hour
        if 5 <= hour < 12:
            part = "Good morning"
        elif 12 <= hour < 17:
            part = "Good afternoon"
        elif 17 <= hour < 21:
            part = "Good evening"
        else:
            part = "Good night"
        return f"{part}, {self._owner}."

    # -- optional LLM summary (non-fatal) ---------------------------------- #
    async def _llm_summary_section(self, briefing: Briefing) -> BriefingSection | None:
        """Optionally add a short natural-language summary; never raise.

        When no ``llm`` is configured this returns ``None`` (no summary). When an
        ``llm`` is present its completion is wrapped in a broad ``except`` so any
        failure — provider error, timeout, empty/blank text — degrades to the
        structured-only briefing rather than failing the whole briefing.
        """
        if self._llm is None:
            return None
        prompt = self._summary_prompt(briefing)
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
            text = (response.text or "").strip()
        except Exception:  # noqa: BLE001 - LLM is optional + non-fatal
            logger.warning("briefing LLM summary failed; using structured-only briefing")
            return None
        if not text:
            return None
        return BriefingSection(title="Summary", items=[text])

    @staticmethod
    def _summary_prompt(briefing: Briefing) -> str:
        """Render the structured briefing into a compact summarization prompt."""
        lines = [briefing.greeting]
        for section in briefing.sections:
            lines.append(section.title + ":")
            lines.extend(f"- {item}" for item in section.items)
        body = "\n".join(lines)
        return (
            "Summarize the following daily briefing in two or three friendly "
            "sentences. Be concise and do not invent details.\n\n" + body
        )
