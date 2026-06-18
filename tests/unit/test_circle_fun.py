"""Unit tests for circle fun: check-in streaks, daily question, stats.

Streaks count consecutive check-in days; the daily question lets members answer a
shared prompt and read each other's answers (consent-gated). All offline; dates
and the reference instant are passed in.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from friday.circle.fun import FunService, InMemoryFunStore
from friday.circle.service import CircleService
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)
TODAY = date(2026, 6, 18)


def _circle() -> CircleService:
    circle = CircleService(InMemoryCircleStore())
    circle.create_group(
        name="Us",
        admin_uid="u-india",
        admin_display_name="Me",
        admin_tz="Asia/Kolkata",
        now=NOW,
        group_id="g1",
    )
    circle.accept_invite(
        code=circle.invite(group_id="g1", by_uid="u-india", now=NOW).code,
        uid="u-us",
        display_name="Bestie",
        tz="America/New_York",
        now=NOW,
    )
    return circle


def _fun(circle: CircleService) -> FunService:
    return FunService(circle, InMemoryFunStore())


def test_check_in_is_idempotent_per_day() -> None:
    fun = _fun(_circle())
    fun.check_in("u-us", on=TODAY)
    fun.check_in("u-us", on=TODAY)
    assert fun.stats("u-us", today=TODAY).total_check_ins == 1


def test_streak_counts_consecutive_days() -> None:
    fun = _fun(_circle())
    for day in (date(2026, 6, 16), date(2026, 6, 17), date(2026, 6, 18)):
        fun.check_in("u-us", on=day)
    assert fun.streak("u-us", today=TODAY) == 3


def test_streak_breaks_on_a_gap() -> None:
    fun = _fun(_circle())
    for day in (date(2026, 6, 14), date(2026, 6, 17), date(2026, 6, 18)):
        fun.check_in("u-us", on=day)
    assert fun.streak("u-us", today=TODAY) == 2


def test_streak_counts_through_yesterday_if_not_yet_today() -> None:
    fun = _fun(_circle())
    for day in (date(2026, 6, 16), date(2026, 6, 17)):
        fun.check_in("u-us", on=day)
    # Hasn't checked in on the 18th yet, but the run through the 17th still counts.
    assert fun.streak("u-us", today=TODAY) == 2


def test_streak_is_zero_when_stale() -> None:
    fun = _fun(_circle())
    fun.check_in("u-us", on=date(2026, 6, 10))
    assert fun.streak("u-us", today=TODAY) == 0


def test_daily_question_and_consent_gated_answers() -> None:
    circle = _circle()
    fun = _fun(circle)
    fun.set_question(TODAY, "Best memory this week?")
    assert fun.question_for(TODAY) == "Best memory this week?"
    fun.answer("u-us", on=TODAY, text="the call last night", now=NOW)
    fun.answer("u-india", on=TODAY, text="your voice note", now=NOW)
    # A circle member sees both answers...
    seen = {a.uid for a in fun.answers_for("u-india", on=TODAY)}
    assert seen == {"u-us", "u-india"}
    # ...a stranger sees none.
    assert fun.answers_for("u-stranger", on=TODAY) == []


def test_stats_summarises_activity() -> None:
    fun = _fun(_circle())
    fun.check_in("u-us", on=date(2026, 6, 17))
    fun.check_in("u-us", on=date(2026, 6, 18))
    fun.answer("u-us", on=TODAY, text="hi", now=NOW)
    stats = fun.stats("u-us", today=TODAY)
    assert stats.total_check_ins == 2
    assert stats.streak == 2
    assert stats.answers == 1
