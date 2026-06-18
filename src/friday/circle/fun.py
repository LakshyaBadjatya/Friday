"""Fun: check-in streaks, a shared daily question, and lightweight stats.

Members check in each day (streaks count consecutive days), answer a shared daily
question (reading each other's answers is consent-gated by
:meth:`CircleService.shares_group`), and can see a small activity summary. Storage
is behind :class:`FunStore`; the in-memory implementation backs tests and local
runs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel

from friday.circle.service import CircleService


class DailyAnswer(BaseModel):
    """One member's answer to a given day's question."""

    id: str
    uid: str
    on: date
    text: str
    created_at: datetime


class CheckinStats(BaseModel):
    """A small activity summary for one member."""

    total_check_ins: int
    streak: int
    answers: int


class FunStore(Protocol):
    """Persistence for check-ins, prompts, and answers."""

    def add_checkin(self, uid: str, on: date) -> None: ...

    def checkin_dates(self, uid: str) -> set[date]: ...

    def set_prompt(self, on: date, text: str) -> None: ...

    def get_prompt(self, on: date) -> str | None: ...

    def save_answer(self, answer: DailyAnswer) -> None: ...

    def list_answers(self, on: date) -> list[DailyAnswer]: ...

    def all_answers(self) -> list[DailyAnswer]: ...


class InMemoryFunStore:
    """A dict-backed :class:`FunStore` for tests and local use."""

    def __init__(self) -> None:
        self._checkins: dict[str, set[date]] = {}
        self._prompts: dict[date, str] = {}
        self._answers: list[DailyAnswer] = []

    def add_checkin(self, uid: str, on: date) -> None:
        self._checkins.setdefault(uid, set()).add(on)

    def checkin_dates(self, uid: str) -> set[date]:
        return set(self._checkins.get(uid, set()))

    def set_prompt(self, on: date, text: str) -> None:
        self._prompts[on] = text

    def get_prompt(self, on: date) -> str | None:
        return self._prompts.get(on)

    def save_answer(self, answer: DailyAnswer) -> None:
        self._answers.append(answer)

    def list_answers(self, on: date) -> list[DailyAnswer]:
        return [a for a in self._answers if a.on == on]

    def all_answers(self) -> list[DailyAnswer]:
        return list(self._answers)


class FunService:
    """Check-ins/streaks, the daily question, and stats for the circle."""

    def __init__(self, circle: CircleService, store: FunStore) -> None:
        self._circle = circle
        self._store = store

    def check_in(self, uid: str, *, on: date) -> None:
        """Record a check-in for ``uid`` on ``on`` (idempotent per day)."""
        self._store.add_checkin(uid, on)

    def streak(self, uid: str, *, today: date) -> int:
        """Consecutive check-in days ending today (or yesterday if not yet today)."""
        dates = self._store.checkin_dates(uid)
        if today in dates:
            cursor = today
        elif (today - timedelta(days=1)) in dates:
            cursor = today - timedelta(days=1)
        else:
            return 0
        count = 0
        while cursor in dates:
            count += 1
            cursor -= timedelta(days=1)
        return count

    def set_question(self, on: date, text: str) -> None:
        self._store.set_prompt(on, text)

    def question_for(self, on: date) -> str | None:
        return self._store.get_prompt(on)

    def answer(self, uid: str, *, on: date, text: str, now: datetime) -> DailyAnswer:
        answer = DailyAnswer(id=uuid4().hex, uid=uid, on=on, text=text, created_at=now)
        self._store.save_answer(answer)
        return answer

    def answers_for(self, viewer_uid: str, *, on: date) -> list[DailyAnswer]:
        """The day's answers from people the viewer shares a group with."""
        return [
            a
            for a in self._store.list_answers(on)
            if self._circle.shares_group(viewer_uid, a.uid)
        ]

    def stats(self, uid: str, *, today: date) -> CheckinStats:
        answers = sum(1 for a in self._store.all_answers() if a.uid == uid)
        return CheckinStats(
            total_check_ins=len(self._store.checkin_dates(uid)),
            streak=self.streak(uid, today=today),
            answers=answers,
        )
