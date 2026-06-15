"""Unit tests for the auto-journaling service + store (Tier 2).

All offline and clock-injected: the :class:`JournalService.build_entry(day)` unit
is driven entirely by the passed ``day`` (the entry ``date`` and every "today"
comparison derive from it). Stores are ephemeral ``":memory:"`` SQLite or
in-process observability objects; the LLM is a scripted
:class:`~friday.providers.llm.FakeLLM` or a deliberately-raising stub. No
network, no key.

Covered:
* Seeded audit rows aggregate into ``event_count`` + deterministic ``highlights``.
* A reminder completed on the day is counted; an undated/other-day one is not.
* The metrics line reflects the snapshot; absent metrics omit it.
* An LLM that raises (or returns blank) still yields a valid structured entry
  (non-fatal), and a working LLM narrates the summary.
* The store ``save``/``get``/``list_entries`` round-trips and upserts by date.
"""

from __future__ import annotations

from datetime import UTC, datetime

from friday.journal.service import JournalEntry, JournalService
from friday.journal.store import SQLiteJournalStore
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.providers.llm import FakeLLM, LLMResponse, Message, ToolSpec
from friday.reminders.store import SQLiteReminderStore

_DAY = datetime(2026, 6, 15, 18, 0, tzinfo=UTC)


def _audit() -> AuditLog:
    """A fresh audit log with a fixed clock (deterministic timestamps)."""
    return AuditLog(clock=lambda: 1.0)


def _reminder_store() -> SQLiteReminderStore:
    """A fresh ephemeral reminder store with a fixed clock (deterministic)."""
    return SQLiteReminderStore(":memory:", clock=lambda: 0.0)


# --------------------------------------------------------------------------- #
# Aggregation: audit rows -> event_count + highlights
# --------------------------------------------------------------------------- #
async def test_build_entry_aggregates_audit_rows_into_count_and_highlights() -> None:
    audit = _audit()
    audit.record(
        correlation_id="c1", tool="web_search", args={}, ok=True, error_code=None
    )
    audit.record(
        correlation_id="c2", tool="notify", args={}, ok=False, error_code="boom"
    )
    service = JournalService(audit, owner_address="Boss")

    entry = await service.build_entry(_DAY)

    assert isinstance(entry, JournalEntry)
    assert entry.date == "2026-06-15"
    assert entry.event_count == 2
    # The count line is deterministic and present.
    assert any("2 tool calls today." == line for line in entry.highlights)
    # Each row is rendered deterministically as tool -> outcome.
    assert any("web_search -> ok" == line for line in entry.highlights)
    assert any("notify -> boom" == line for line in entry.highlights)


async def test_build_entry_empty_day_is_zero_events() -> None:
    service = JournalService(_audit(), owner_address="Boss")
    entry = await service.build_entry(_DAY)
    assert entry.event_count == 0
    assert any("0 tool calls today." == line for line in entry.highlights)


async def test_build_entry_is_deterministic_for_same_seed() -> None:
    audit = _audit()
    audit.record(
        correlation_id="c1", tool="home", args={}, ok=True, error_code=None
    )
    service = JournalService(audit, owner_address="Boss")

    first = await service.build_entry(_DAY)
    second = await service.build_entry(_DAY)
    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------- #
# Reminders completed that day
# --------------------------------------------------------------------------- #
async def test_completed_reminder_on_day_is_counted() -> None:
    store = _reminder_store()
    reminder = store.add("ship the build", due_at="2026-06-15T09:00:00+00:00")
    store.complete(reminder.id)  # one-shot -> done on the journaled day
    service = JournalService(_audit(), reminder_store=store, owner_address="Boss")

    entry = await service.build_entry(_DAY)

    assert any("1 reminder completed today." == line for line in entry.highlights)


async def test_reminder_completed_on_other_day_not_counted() -> None:
    store = _reminder_store()
    reminder = store.add("old task", due_at="2026-06-10T09:00:00+00:00")
    store.complete(reminder.id)  # done, but due_at falls on a different day
    service = JournalService(_audit(), reminder_store=store, owner_address="Boss")

    entry = await service.build_entry(_DAY)

    assert any("0 reminders completed today." == line for line in entry.highlights)


async def test_no_reminder_store_skips_reminder_line_gracefully() -> None:
    service = JournalService(_audit(), owner_address="Boss")
    entry = await service.build_entry(_DAY)
    assert not any("completed today" in line for line in entry.highlights)


class _OpaqueReminderStore:
    """A reminder store that does not expose a way to enumerate completions."""


async def test_store_without_list_reminders_skips_line() -> None:
    service = JournalService(
        _audit(), reminder_store=_OpaqueReminderStore(), owner_address="Boss"
    )
    entry = await service.build_entry(_DAY)  # must not raise
    assert not any("completed today" in line for line in entry.highlights)


# --------------------------------------------------------------------------- #
# Metrics glance
# --------------------------------------------------------------------------- #
async def test_metrics_line_reflects_snapshot() -> None:
    metrics = Metrics()
    metrics.inc_requests(4)
    metrics.inc_tool_calls(2)
    metrics.inc_errors(1)
    service = JournalService(_audit(), metrics=metrics, owner_address="Boss")

    entry = await service.build_entry(_DAY)

    metric_lines = [line for line in entry.highlights if line.startswith("Metrics:")]
    assert len(metric_lines) == 1
    line = metric_lines[0]
    assert "4 requests" in line
    assert "2 tool calls" in line
    assert "1 errors" in line


async def test_no_metrics_omits_metrics_line() -> None:
    service = JournalService(_audit(), owner_address="Boss")
    entry = await service.build_entry(_DAY)
    assert not any(line.startswith("Metrics:") for line in entry.highlights)


# --------------------------------------------------------------------------- #
# Summary: deterministic + optional LLM narration (NON-FATAL)
# --------------------------------------------------------------------------- #
async def test_no_llm_uses_deterministic_summary() -> None:
    service = JournalService(_audit(), owner_address="Tony")
    entry = await service.build_entry(_DAY)
    assert "2026-06-15" in entry.summary
    assert "Tony" in entry.summary


async def test_working_llm_narrates_summary() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="A productive, calm Monday.")])
    service = JournalService(_audit(), llm=llm, owner_address="Boss")
    entry = await service.build_entry(_DAY)
    assert entry.summary == "A productive, calm Monday."


class _RaisingLLM:
    """An LLM stub whose ``complete`` always raises (to test the fallback)."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("llm exploded")


async def test_llm_error_yields_valid_structured_entry() -> None:
    audit = _audit()
    audit.record(
        correlation_id="c1", tool="web_search", args={}, ok=True, error_code=None
    )
    # Cast through the constructor: the stub satisfies the structural contract.
    service = JournalService(audit, llm=_RaisingLLM(), owner_address="Boss")  # type: ignore[arg-type]

    entry = await service.build_entry(_DAY)  # must not raise

    assert isinstance(entry, JournalEntry)
    assert entry.event_count == 1
    # Falls back to the deterministic structured summary.
    assert "2026-06-15" in entry.summary
    assert any("web_search -> ok" == line for line in entry.highlights)


async def test_blank_llm_text_falls_back_to_structured_summary() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="   ")])
    service = JournalService(_audit(), llm=llm, owner_address="Boss")
    entry = await service.build_entry(_DAY)
    assert "2026-06-15" in entry.summary  # deterministic fallback


async def test_build_entry_uses_injected_day_not_wallclock() -> None:
    service = JournalService(_audit(), owner_address="Boss")
    other_day = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    entry = await service.build_entry(other_day)
    assert entry.date == "2025-01-02"


# --------------------------------------------------------------------------- #
# Store: save / get / list round-trip + upsert by date
# --------------------------------------------------------------------------- #
def _entry(date: str, *, summary: str = "s", events: int = 1) -> JournalEntry:
    return JournalEntry(
        date=date, summary=summary, highlights=["a", "b"], event_count=events
    )


def test_store_save_get_roundtrip() -> None:
    store = SQLiteJournalStore(":memory:")
    saved = store.save(_entry("2026-06-15", summary="day one", events=3))
    assert saved.date == "2026-06-15"

    fetched = store.get("2026-06-15")
    assert fetched is not None
    assert fetched.summary == "day one"
    assert fetched.event_count == 3
    assert fetched.highlights == ["a", "b"]


def test_store_get_missing_is_none() -> None:
    store = SQLiteJournalStore(":memory:")
    assert store.get("2026-01-01") is None


def test_store_upsert_by_date_overwrites() -> None:
    store = SQLiteJournalStore(":memory:")
    store.save(_entry("2026-06-15", summary="first", events=1))
    store.save(_entry("2026-06-15", summary="second", events=9))

    fetched = store.get("2026-06-15")
    assert fetched is not None
    assert fetched.summary == "second"
    assert fetched.event_count == 9
    # Only one row for the date — no duplicate.
    assert len(store.list_entries()) == 1


def test_store_list_entries_most_recent_first() -> None:
    store = SQLiteJournalStore(":memory:")
    store.save(_entry("2026-06-13"))
    store.save(_entry("2026-06-15"))
    store.save(_entry("2026-06-14"))

    entries = store.list_entries()
    assert [e.date for e in entries] == ["2026-06-15", "2026-06-14", "2026-06-13"]


def test_store_list_entries_respects_limit() -> None:
    store = SQLiteJournalStore(":memory:")
    for day in range(10, 20):
        store.save(_entry(f"2026-06-{day}"))
    assert len(store.list_entries(limit=3)) == 3


def test_store_file_path_roundtrip(tmp_path: object) -> None:
    """A file-backed store (connection-per-call) round-trips across instances."""
    from pathlib import Path

    db = Path(str(tmp_path)) / "journal.db"
    store = SQLiteJournalStore(str(db))
    store.save(_entry("2026-06-15", summary="persisted", events=7))

    # A fresh instance over the same file sees the saved entry.
    reopened = SQLiteJournalStore(str(db))
    fetched = reopened.get("2026-06-15")
    assert fetched is not None
    assert fetched.summary == "persisted"
    assert fetched.event_count == 7
