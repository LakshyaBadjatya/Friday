# © Lakshya Badjatya — Author
"""Unit tests for the per-turn budgeter (``friday.models.budget``).

The budgeter is pure arithmetic: a :class:`TurnBudget` accumulates spend against
token (and optional dollar) caps, and a :class:`Budgeter` keeps one such tally
per session. These tests pin the spend math, the inclusive ``>=`` over-budget
boundary, the downshift threshold crossing, multi-session isolation, and the
``start_turn`` reset — all deterministic, no network, no LLM SDK, no clock.
"""

from __future__ import annotations

import pytest

from friday.models.budget import Budgeter, TurnBudget


# --------------------------------------------------------------------------- #
# TurnBudget: record / remaining math
# --------------------------------------------------------------------------- #
def test_record_accumulates_tokens_and_usd() -> None:
    budget = TurnBudget(max_tokens=1000, max_usd=1.0)
    budget.record(100, 0.05)
    budget.record(250, 0.10)
    assert budget.spent_tokens == 350
    # Float accumulation: compare with tolerance, not exact equality.
    assert budget.spent_usd == pytest.approx(0.15)


def test_remaining_tokens_subtracts_spend() -> None:
    budget = TurnBudget(max_tokens=1000)
    budget.record(300)
    assert budget.remaining_tokens() == 700


def test_remaining_tokens_clamps_at_zero() -> None:
    budget = TurnBudget(max_tokens=1000)
    budget.record(1500)
    # Never negative even when overspent.
    assert budget.remaining_tokens() == 0


# --------------------------------------------------------------------------- #
# TurnBudget: over_budget boundary (>= not >)
# --------------------------------------------------------------------------- #
def test_over_budget_false_below_token_cap() -> None:
    budget = TurnBudget(max_tokens=1000)
    budget.record(999)
    assert budget.over_budget() is False


def test_over_budget_true_exactly_at_token_cap() -> None:
    budget = TurnBudget(max_tokens=1000)
    budget.record(1000)
    # Inclusive boundary: landing exactly on the cap is over budget.
    assert budget.over_budget() is True


def test_over_budget_true_above_token_cap() -> None:
    budget = TurnBudget(max_tokens=1000)
    budget.record(1001)
    assert budget.over_budget() is True


def test_over_budget_ignores_usd_when_cap_unset() -> None:
    budget = TurnBudget(max_tokens=1000, max_usd=None)
    budget.record(10, 999.0)
    # No dollar cap -> dollars never trip over_budget; tokens are well under.
    assert budget.over_budget() is False


def test_over_budget_true_exactly_at_usd_cap() -> None:
    budget = TurnBudget(max_tokens=1_000_000, max_usd=0.50)
    budget.record(10, 0.50)
    # Tokens nowhere near the cap, but the dollar cap is reached (inclusive).
    assert budget.over_budget() is True


def test_over_budget_false_below_usd_cap() -> None:
    budget = TurnBudget(max_tokens=1_000_000, max_usd=0.50)
    budget.record(10, 0.49)
    assert budget.over_budget() is False


# --------------------------------------------------------------------------- #
# Budgeter: record / remaining delegation
# --------------------------------------------------------------------------- #
def test_budgeter_record_and_remaining() -> None:
    budgeter = Budgeter(max_tokens=1000)
    budgeter.record("s1", 400)
    assert budgeter.remaining("s1") == 600


def test_budgeter_record_lazily_starts_first_turn() -> None:
    budgeter = Budgeter(max_tokens=1000)
    # No explicit start_turn first — record on an unseen session is fine.
    budgeter.record("fresh", 100)
    assert budgeter.remaining("fresh") == 900


def test_budgeter_remaining_full_for_unseen_session() -> None:
    budgeter = Budgeter(max_tokens=1000)
    # Querying an unseen session lazily starts a full turn.
    assert budgeter.remaining("never-touched") == 1000


# --------------------------------------------------------------------------- #
# Budgeter: should_downshift threshold crossing
# --------------------------------------------------------------------------- #
def test_should_downshift_false_below_threshold() -> None:
    budgeter = Budgeter(max_tokens=1000, downshift_at=0.8)
    budgeter.record("s1", 799)
    # 799 < 0.8 * 1000 == 800.
    assert budgeter.should_downshift("s1") is False


def test_should_downshift_true_at_threshold() -> None:
    budgeter = Budgeter(max_tokens=1000, downshift_at=0.8)
    budgeter.record("s1", 800)
    # 800 >= 0.8 * 1000 -> trip (inclusive).
    assert budgeter.should_downshift("s1") is True


def test_should_downshift_true_above_threshold() -> None:
    budgeter = Budgeter(max_tokens=1000, downshift_at=0.8)
    budgeter.record("s1", 950)
    assert budgeter.should_downshift("s1") is True


def test_should_downshift_trips_on_usd_cap() -> None:
    budgeter = Budgeter(max_tokens=1_000_000, max_usd=1.0, downshift_at=0.8)
    budgeter.record("s1", 10, 1.0)
    # Tokens nowhere near the threshold, but the dollar cap is reached.
    assert budgeter.should_downshift("s1") is True


def test_should_downshift_softer_than_over_budget() -> None:
    budgeter = Budgeter(max_tokens=1000, downshift_at=0.8)
    budget = budgeter.start_turn("s1")
    budgeter.record("s1", 850)
    # Downshift trips (>= 800) before the turn is over budget (< 1000).
    assert budgeter.should_downshift("s1") is True
    assert budget.over_budget() is False


# --------------------------------------------------------------------------- #
# Budgeter: multi-session isolation
# --------------------------------------------------------------------------- #
def test_sessions_do_not_share_budget() -> None:
    budgeter = Budgeter(max_tokens=1000)
    budgeter.record("a", 900)
    budgeter.record("b", 100)
    assert budgeter.remaining("a") == 100
    assert budgeter.remaining("b") == 900
    assert budgeter.should_downshift("a") is True
    assert budgeter.should_downshift("b") is False


# --------------------------------------------------------------------------- #
# Budgeter: start_turn reset
# --------------------------------------------------------------------------- #
def test_start_turn_resets_session_spend() -> None:
    budgeter = Budgeter(max_tokens=1000)
    budgeter.record("s1", 900)
    assert budgeter.remaining("s1") == 100
    fresh = budgeter.start_turn("s1")
    # The new turn is zeroed and carries the configured caps.
    assert fresh.spent_tokens == 0
    assert fresh.spent_usd == 0.0
    assert fresh.max_tokens == 1000
    assert budgeter.remaining("s1") == 1000
    assert budgeter.should_downshift("s1") is False


def test_start_turn_returns_the_live_budget() -> None:
    budgeter = Budgeter(max_tokens=1000)
    budget = budgeter.start_turn("s1")
    budgeter.record("s1", 250)
    # The returned object is the same one record() mutates.
    assert budget.spent_tokens == 250
