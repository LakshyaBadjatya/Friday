"""Unit tests for spoken reminders — :mod:`friday.siri.when` and ``siri.tasks``.

Every test injects a fixed ``now`` (Saturday 15 August 2026, 15:30 IST) rather
than reading the clock, so "tomorrow at 9" means one exact instant and the suite
cannot pass in the morning and fail in the evening.

The parsing assertions are the important ones: a reminder stored against a
misheard time is worse than one that was never stored, and the spoken read-back
is the user's only chance to catch it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from friday.reminders.store import SQLiteReminderStore
from friday.siri.tasks import handle
from friday.siri.when import parse_when, strip_time_words

#: Saturday 15 Aug 2026, 10:00 UTC == 15:30 Asia/Kolkata.
NOW = datetime(2026, 8, 15, 10, 0, tzinfo=UTC)
TZ = "Asia/Kolkata"


@pytest.fixture
def store() -> SQLiteReminderStore:
    made = SQLiteReminderStore(":memory:")
    made.init_schema()
    return made


# --------------------------------------------------------------------------- #
# Time parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # A bare hour leans waking: "at 6" is the evening, not dawn.
        ("call mum at 6", "today at 6 PM"),
        # ...but an explicit meridiem always wins, and 6 AM today has passed.
        ("call mum at 6 am", "tomorrow at 6 AM"),
        ("wake me at 7", "tomorrow at 7 AM"),
        # Relative offsets.
        ("in 10 minutes", "today at 3:40 PM"),
        ("in an hour", "today at 4:30 PM"),
        ("in 2 days", "Monday at 3:30 PM"),
        # Named days.
        ("tomorrow at 9", "tomorrow at 9 AM"),
        ("on friday at 5 pm", "Friday at 5 PM"),
        # Parts of the day imply an hour.
        ("tomorrow morning", "tomorrow at 9 AM"),
        ("tonight", "today at 8 PM"),
        # A time already gone rolls forward rather than firing in the past.
        ("at 9", "tomorrow at 9 AM"),
    ],
)
def test_parse_when_resolves_spoken_times(spoken: str, expected: str) -> None:
    parsed = parse_when(spoken, tz_name=TZ, now=NOW)
    assert parsed is not None, spoken
    assert parsed[1] == expected


def test_parse_when_returns_none_without_a_time() -> None:
    """An undated reminder is valid, not an error."""
    assert parse_when("buy milk", tz_name=TZ, now=NOW) is None


def test_parse_when_stores_utc() -> None:
    """The store holds UTC; only the spoken half is local."""
    parsed = parse_when("at 6", tz_name=TZ, now=NOW)
    assert parsed is not None
    stored = datetime.fromisoformat(parsed[0])
    assert stored.utcoffset() is not None
    assert stored.utcoffset().total_seconds() == 0


def test_unknown_timezone_degrades_to_utc_rather_than_failing() -> None:
    assert parse_when("at 6", tz_name="Mars/Olympus", now=NOW) is not None


def test_strip_time_words_leaves_only_the_task() -> None:
    assert strip_time_words("call mum at 6 tomorrow") == "call mum"
    assert strip_time_words("submit the form in 2 days") == "submit the form"
    assert strip_time_words("buy milk") == "buy milk"


# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #
def test_create_stores_and_confirms_both_halves(store: SQLiteReminderStore) -> None:
    reply = handle(store, "remind me to call mum at 6", NOW, tz_name=TZ)

    assert reply == "Got it, Boss — call mum, today at 6 PM."
    (stored,) = store.list_reminders()
    assert stored.text == "call mum"
    assert stored.due_at is not None


def test_create_keeps_the_wording_verbatim(store: SQLiteReminderStore) -> None:
    """The task text must not be paraphrased — only the time words come out."""
    handle(store, "remind me to email the landlord about the deposit", NOW, tz_name=TZ)

    (stored,) = store.list_reminders()
    assert stored.text == "email the landlord about the deposit"


def test_create_captures_recurrence_without_repeating_it(
    store: SQLiteReminderStore,
) -> None:
    reply = handle(store, "remind me to take medicine every day at 9 am", NOW, tz_name=TZ)

    assert reply == "Got it, Boss — take medicine, tomorrow at 9 AM, every day."
    (stored,) = store.list_reminders()
    assert stored.recurrence == "daily"
    assert "every day" not in stored.text  # it lives in the column, not the text


def test_create_without_a_time_says_so(store: SQLiteReminderStore) -> None:
    reply = handle(store, "remind me to buy milk", NOW, tz_name=TZ)

    assert reply is not None
    assert "buy milk" in reply
    assert store.list_reminders()[0].due_at is None


def test_list_reads_back_what_is_open(store: SQLiteReminderStore) -> None:
    handle(store, "remind me to call mum at 6", NOW, tz_name=TZ)
    handle(store, "remind me to submit the form tomorrow", NOW, tz_name=TZ)

    reply = handle(store, "what are my reminders", NOW, tz_name=TZ)

    assert reply is not None
    assert "2 reminders" in reply
    assert "call mum" in reply
    assert "submit the form" in reply


def test_list_when_empty_is_honest(store: SQLiteReminderStore) -> None:
    assert handle(store, "what are my reminders", NOW, tz_name=TZ) == (
        "Nothing on your list, Boss."
    )


def test_complete_closes_the_matching_reminder(store: SQLiteReminderStore) -> None:
    handle(store, "remind me to call mum at 6", NOW, tz_name=TZ)

    reply = handle(store, "i finished call mum", NOW, tz_name=TZ)

    assert reply is not None
    assert "off your list" in reply
    assert handle(store, "what are my reminders", NOW, tz_name=TZ) == (
        "Nothing on your list, Boss."
    )


def test_complete_an_unknown_task_says_so(store: SQLiteReminderStore) -> None:
    reply = handle(store, "i finished washing the car", NOW, tz_name=TZ)

    assert reply is not None
    assert "don't have an open reminder" in reply


@pytest.mark.parametrize(
    "query",
    [
        "what is the capital of france",
        "how far is bhopal",
        "what was the last topic",
        "12 * 37",
    ],
)
def test_non_reminder_queries_fall_through(
    store: SQLiteReminderStore, query: str
) -> None:
    assert handle(store, query, NOW, tz_name=TZ) is None


def test_missing_store_falls_through() -> None:
    """No reminder store wired: skip the branch rather than fail the turn."""
    assert handle(None, "remind me to call mum at 6", NOW, tz_name=TZ) is None


def test_storage_failure_is_spoken_not_raised() -> None:
    class _Exploding:
        def add(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("disk full")

    reply = handle(_Exploding(), "remind me to call mum at 6", NOW, tz_name=TZ)

    assert reply is not None
    assert "couldn't save" in reply
