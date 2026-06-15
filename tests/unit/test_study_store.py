"""Unit tests for :class:`friday.study.store.SQLiteStudyStore` (Tier 2 study).

The study store is the local-first, SQLite-backed durable layer for flashcards
and logged study sessions. Every test here is offline, uses a *tmp-file*
database (so the connection-per-call path is exercised and the store is
thread-safe by construction), and injects a deterministic ``clock`` so timestamps
(``due_at`` on review, ``at`` on a session) never read the wall clock.

Pinned behaviours:

* ``add_card`` returns a typed :class:`Flashcard` with an assigned id, the SM-2
  starting ease, and ``due_at`` unset (a brand-new card is immediately due);
  the schema init is idempotent.
* ``due_cards(now)`` returns only cards whose ``due_at`` is at or before the
  passed ``now`` (plus never-reviewed cards) — driven by the passed timestamp.
* ``review_card(id, grade)`` applies SM-2, persists the new ease/interval/reps,
  and reschedules ``due_at`` to ``now + interval_days`` (now from the clock).
* ``list_cards`` filters by deck; ``delete_card`` removes a row.
* sessions round-trip: ``add_session`` stamps ``at`` from the clock and
  ``list_sessions`` returns them most-recent first, capped at ``limit``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from friday.study.store import Flashcard, SQLiteStudyStore, StudySession


def _store(tmp_path: Path, *, now: datetime | None = None) -> SQLiteStudyStore:
    """A tmp-file store with a fixed injected clock (no wall-clock reads)."""
    fixed = now or datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    return SQLiteStudyStore(str(tmp_path / "study.db"), clock=lambda: fixed)


# --------------------------------------------------------------------------- #
# add_card + schema
# --------------------------------------------------------------------------- #
def test_add_card_returns_typed_card_with_defaults(tmp_path: Path) -> None:
    store = _store(tmp_path)
    card = store.add_card("french", "bonjour", "hello")

    assert isinstance(card, Flashcard)
    assert card.id >= 1
    assert card.deck == "french"
    assert card.front == "bonjour"
    assert card.back == "hello"
    assert card.ease == 2.5
    assert card.interval_days == 0
    assert card.reps == 0
    assert card.due_at is None  # brand-new card is immediately due


def test_schema_init_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "study.db")
    fixed = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    first = SQLiteStudyStore(db, clock=lambda: fixed)
    first.add_card("d", "f", "b")
    # Re-opening the same file must not clobber existing rows.
    second = SQLiteStudyStore(db, clock=lambda: fixed)
    assert [c.front for c in second.list_cards()] == ["f"]


# --------------------------------------------------------------------------- #
# list_cards + delete_card
# --------------------------------------------------------------------------- #
def test_list_cards_filters_by_deck(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_card("french", "bonjour", "hello")
    store.add_card("french", "merci", "thanks")
    store.add_card("spanish", "hola", "hi")

    french = [c.front for c in store.list_cards(deck="french")]
    assert set(french) == {"bonjour", "merci"}
    spanish = [c.front for c in store.list_cards(deck="spanish")]
    assert spanish == ["hola"]
    all_cards = store.list_cards()
    assert len(all_cards) == 3


def test_delete_card_removes_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    card = store.add_card("d", "f", "b")
    assert store.delete_card(card.id) == 1
    assert store.list_cards() == []
    # Deleting again is a no-op.
    assert store.delete_card(card.id) == 0


# --------------------------------------------------------------------------- #
# due_cards filters by the passed now
# --------------------------------------------------------------------------- #
def test_due_cards_includes_never_reviewed_cards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_card("d", "f", "b")
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    due = store.due_cards(now)
    assert [c.front for c in due] == ["f"]


def test_due_cards_filters_by_passed_now(tmp_path: Path) -> None:
    # Review a card so it gets a future due_at, then assert it is excluded until
    # that due date is reached.
    review_time = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    store = _store(tmp_path, now=review_time)
    card = store.add_card("d", "f", "b")
    reviewed = store.review_card(card.id, 4)  # first pass -> interval 1 day
    assert reviewed.due_at is not None

    # Before the due date: not due.
    before = review_time + timedelta(hours=12)
    assert store.due_cards(before) == []
    # At/after the due date: due again.
    at_due = review_time + timedelta(days=1)
    assert [c.front for c in store.due_cards(at_due)] == ["f"]


# --------------------------------------------------------------------------- #
# review_card advances the card + reschedules due_at
# --------------------------------------------------------------------------- #
def test_review_card_advances_and_reschedules(tmp_path: Path) -> None:
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    store = _store(tmp_path, now=now)
    card = store.add_card("d", "f", "b")

    reviewed = store.review_card(card.id, 4)
    # First pass -> interval 1 day, reps 1.
    assert reviewed.reps == 1
    assert reviewed.interval_days == 1
    assert reviewed.due_at == (now + timedelta(days=1)).isoformat()

    # The change is persisted (re-read it).
    again = store.list_cards(deck="d")[0]
    assert again.reps == 1
    assert again.interval_days == 1
    assert again.due_at == (now + timedelta(days=1)).isoformat()


def test_review_card_progression_matches_sm2(tmp_path: Path) -> None:
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    store = _store(tmp_path, now=now)
    card = store.add_card("d", "f", "b")

    first = store.review_card(card.id, 4)
    second = store.review_card(card.id, 4)
    assert first.interval_days == 1
    assert second.interval_days == 6
    assert second.reps == 2
    assert second.due_at == (now + timedelta(days=6)).isoformat()


def test_review_unknown_card_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.review_card(9999, 4) is None


# --------------------------------------------------------------------------- #
# sessions round-trip
# --------------------------------------------------------------------------- #
def test_add_session_stamps_at_from_clock(tmp_path: Path) -> None:
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    store = _store(tmp_path, now=now)
    session = store.add_session("calculus", 45)

    assert isinstance(session, StudySession)
    assert session.id >= 1
    assert session.topic == "calculus"
    assert session.minutes == 45
    assert session.at == now.isoformat()


def test_list_sessions_most_recent_first_and_capped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_session("a", 10)
    store.add_session("b", 20)
    store.add_session("c", 30)

    listed = store.list_sessions(limit=2)
    # Most-recent (highest id) first, capped at the limit.
    assert [s.topic for s in listed] == ["c", "b"]


def test_list_sessions_round_trips_all(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_session("a", 10)
    store.add_session("b", 20)
    topics = {s.topic for s in store.list_sessions(limit=50)}
    assert topics == {"a", "b"}
