"""Unit tests for the proactive briefing service (Tier 1).

All offline and clock-injected: the :class:`BriefingService.build(now)` unit is
driven entirely by the passed ``now`` (the reminder buckets, the greeting, and
``generated_at`` all derive from it). Stores are ephemeral ``":memory:"`` SQLite
or in-process observability objects; the LLM is a scripted
:class:`~friday.providers.llm.FakeLLM` or a deliberately-raising stub. No
network, no key.

Covered:
* Seeded reminders (one overdue, one due today, one upcoming) land in the right
  buckets.
* The recent-activity section reflects seeded audit rows; absent audit -> no
  section.
* The greeting varies by time-of-day and addresses the owner.
* The at-a-glance metrics line reflects the metrics snapshot.
* An LLM that raises still yields a valid structured briefing (never raises),
  and a working LLM adds a summary section.
"""

from __future__ import annotations

from datetime import UTC, datetime

from friday.briefing.service import Briefing, BriefingService
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.providers.llm import FakeLLM, LLMResponse, Message, ToolSpec
from friday.reminders.store import SQLiteReminderStore


def _store() -> SQLiteReminderStore:
    """A fresh ephemeral reminder store with a fixed clock (deterministic)."""
    return SQLiteReminderStore(":memory:", clock=lambda: 0.0)


def _section(briefing: Briefing, title: str) -> list[str]:
    """Return the items of the section titled ``title`` (empty if absent)."""
    for section in briefing.sections:
        if section.title == title:
            return section.items
    return []


def _titles(briefing: Briefing) -> list[str]:
    return [s.title for s in briefing.sections]


# --------------------------------------------------------------------------- #
# Reminder bucketing
# --------------------------------------------------------------------------- #
async def test_reminders_bucketed_overdue_due_today_upcoming() -> None:
    store = _store()
    # "now" is early morning (08:00) on 2026-06-15.
    now = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
    store.add("call dentist", due_at="2026-06-14T09:00:00+00:00")  # overdue (yesterday)
    store.add("standup", due_at="2026-06-15T08:00:00+00:00")  # due today (== now)
    store.add("review PRs", due_at="2026-06-15T03:00:00+00:00")  # due today (passed)
    store.add("daily sync", due_at="2026-06-15T17:00:00+00:00")  # due LATER today
    store.add("file taxes", due_at="2026-06-20T09:00:00+00:00")  # upcoming (next week)

    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)

    overdue = _section(briefing, "Overdue")
    due_today = _section(briefing, "Due today")
    upcoming = _section(briefing, "Upcoming")

    # Yesterday -> Overdue, and only that.
    assert any("call dentist" in line for line in overdue)
    assert not any("standup" in line for line in overdue)
    assert not any("daily sync" in line for line in overdue)

    # Every reminder whose calendar date is today lands in Due today —
    # whether the time has passed (review PRs, standup) or is still ahead
    # (daily sync at 17:00 while now is 08:00).
    assert any("standup" in line for line in due_today)
    assert any("review PRs" in line for line in due_today)
    assert any("daily sync" in line for line in due_today)
    assert not any("call dentist" in line for line in due_today)
    assert not any("file taxes" in line for line in due_today)

    # A reminder due later today must NOT leak into Upcoming.
    assert any("file taxes" in line for line in upcoming)
    assert not any("daily sync" in line for line in upcoming)
    assert not any("standup" in line for line in upcoming)


async def test_reminder_due_later_today_lands_in_due_today_not_upcoming() -> None:
    store = _store()
    # now is 08:00; the reminder is due at 17:00 the same calendar day.
    now = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
    store.add("evening call", due_at="2026-06-15T17:00:00+00:00")

    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)

    assert any("evening call" in line for line in _section(briefing, "Due today"))
    assert not any(
        "evening call" in line for line in _section(briefing, "Upcoming")
    )
    assert not any(
        "evening call" in line for line in _section(briefing, "Overdue")
    )


async def test_upcoming_section_caps_at_limit() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
    # Seven reminders all dated strictly after today; only five should show.
    for day in range(16, 23):  # 2026-06-16 .. 2026-06-22
        store.add(f"task {day}", due_at=f"2026-06-{day}T09:00:00+00:00")

    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)

    upcoming = _section(briefing, "Upcoming")
    assert len(upcoming) == 5
    # Soonest-due first: the earliest five dates are listed.
    assert any("task 16" in line for line in upcoming)
    assert not any("task 22" in line for line in upcoming)


async def test_empty_reminder_buckets_have_placeholder_lines() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)

    # All three reminder sections are always present.
    assert "Overdue" in _titles(briefing)
    assert "Due today" in _titles(briefing)
    assert "Upcoming" in _titles(briefing)
    # Each carries a single placeholder line.
    assert _section(briefing, "Overdue") == ["Nothing overdue."]
    assert _section(briefing, "Upcoming") == ["Nothing upcoming."]


async def test_undated_reminders_never_bucketed() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    store.add("buy milk")  # no due date

    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)

    assert not any(
        "buy milk" in line
        for title in ("Overdue", "Due today", "Upcoming")
        for line in _section(briefing, title)
    )


# --------------------------------------------------------------------------- #
# Recent activity (audit log)
# --------------------------------------------------------------------------- #
async def test_recent_activity_reflects_seeded_audit_rows() -> None:
    store = _store()
    audit = AuditLog(clock=lambda: 1.0)
    audit.record(
        correlation_id="c1", tool="web_search", args={}, ok=True, error_code=None
    )
    audit.record(
        correlation_id="c2",
        tool="notify",
        args={},
        ok=False,
        error_code="boom",
    )
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(store, audit_log=audit, owner_address="Boss")
    briefing = await service.build(now)

    activity = _section(briefing, "Recent activity")
    assert any("web_search" in line for line in activity)
    assert any("notify" in line and "boom" in line for line in activity)


async def test_recent_activity_respects_limit() -> None:
    store = _store()
    audit = AuditLog(clock=lambda: 1.0)
    for i in range(10):
        audit.record(
            correlation_id=f"c{i}",
            tool=f"tool_{i}",
            args={},
            ok=True,
            error_code=None,
        )
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(
        store, audit_log=audit, owner_address="Boss", recent_activity_limit=3
    )
    briefing = await service.build(now)

    activity = _section(briefing, "Recent activity")
    assert len(activity) == 3
    # Most recent rows, oldest-first within the window.
    assert "tool_9" in activity[-1]


async def test_no_audit_log_omits_recent_activity_section() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)
    assert "Recent activity" not in _titles(briefing)


# --------------------------------------------------------------------------- #
# At-a-glance metrics
# --------------------------------------------------------------------------- #
async def test_at_a_glance_reflects_metrics_snapshot() -> None:
    store = _store()
    metrics = Metrics()
    metrics.inc_requests(4)
    metrics.inc_tool_calls(2)
    metrics.inc_errors(1)
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(store, metrics=metrics, owner_address="Boss")
    briefing = await service.build(now)

    glance = _section(briefing, "At a glance")
    assert len(glance) == 1
    line = glance[0]
    assert "4 requests" in line
    assert "2 tool calls" in line
    assert "1 errors" in line


async def test_no_metrics_omits_at_a_glance_section() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)
    assert "At a glance" not in _titles(briefing)


# --------------------------------------------------------------------------- #
# Greeting: time-of-day aware + addresses owner
# --------------------------------------------------------------------------- #
async def test_greeting_varies_by_time_of_day_and_addresses_owner() -> None:
    store = _store()
    service = BriefingService(store, owner_address="Tony")

    morning = await service.build(datetime(2026, 6, 15, 8, 0, tzinfo=UTC))
    afternoon = await service.build(datetime(2026, 6, 15, 14, 0, tzinfo=UTC))
    evening = await service.build(datetime(2026, 6, 15, 19, 0, tzinfo=UTC))
    night = await service.build(datetime(2026, 6, 15, 23, 0, tzinfo=UTC))

    assert "morning" in morning.greeting.lower()
    assert "afternoon" in afternoon.greeting.lower()
    assert "evening" in evening.greeting.lower()
    assert "night" in night.greeting.lower()

    # All four greetings differ and every one addresses the owner.
    greetings = {
        morning.greeting,
        afternoon.greeting,
        evening.greeting,
        night.greeting,
    }
    assert len(greetings) == 4
    for greeting in greetings:
        assert "Tony" in greeting


async def test_generated_at_is_the_injected_now() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 9, 30, tzinfo=UTC)
    service = BriefingService(store, owner_address="Boss")
    briefing = await service.build(now)
    assert briefing.generated_at == now.isoformat()


# --------------------------------------------------------------------------- #
# LLM optional + NON-FATAL
# --------------------------------------------------------------------------- #
class _RaisingLLM:
    """An LLM stub whose ``complete`` always raises (to test the fallback)."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("llm exploded")


async def test_llm_error_still_yields_valid_structured_briefing() -> None:
    store = _store()
    store.add("standup", due_at="2026-06-15T08:00:00+00:00")
    now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    # Cast through the constructor: the stub satisfies the structural contract.
    service = BriefingService(store, llm=_RaisingLLM(), owner_address="Boss")  # type: ignore[arg-type]

    briefing = await service.build(now)  # must not raise

    assert isinstance(briefing, Briefing)
    # The structured sections are intact; no summary section was added.
    assert "Due today" in _titles(briefing)
    assert "Summary" not in _titles(briefing)
    assert any("standup" in line for line in _section(briefing, "Due today"))


async def test_working_llm_adds_summary_section() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    llm = FakeLLM(responses=[LLMResponse(text="You have a calm day ahead.")])
    service = BriefingService(store, llm=llm, owner_address="Boss")

    briefing = await service.build(now)

    assert "Summary" in _titles(briefing)
    assert _section(briefing, "Summary") == ["You have a calm day ahead."]


async def test_blank_llm_text_omits_summary_section() -> None:
    store = _store()
    now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    llm = FakeLLM(responses=[LLMResponse(text="   ")])
    service = BriefingService(store, llm=llm, owner_address="Boss")

    briefing = await service.build(now)
    assert "Summary" not in _titles(briefing)
