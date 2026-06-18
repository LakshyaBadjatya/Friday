"""Unit tests for circle voice intents: parsing + handling.

``parse_intent`` recognises the status phrases a Siri caller would say; everything
else returns ``None`` so the route falls through to the general assistant.
``handle_intent`` resolves a spoken name within the caller's circle and either sets
their status or describes the target's — falling through (``None``) when the name
isn't someone in the caller's circle, so general queries reach the assistant.
"""

from __future__ import annotations

from datetime import UTC, datetime

from friday.circle.intents import SetStatus, StatusQuery, handle_intent, parse_intent
from friday.circle.service import CircleService
from friday.circle.status import InMemoryStatusStore, StatusService
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)


# --- parsing -----------------------------------------------------------------
def test_parse_set_status() -> None:
    assert parse_intent("set my status to coding") == SetStatus(text="coding")


def test_parse_set_mood() -> None:
    assert parse_intent("I'm feeling great") == SetStatus(mood="great")
    assert parse_intent("set my mood to tired") == SetStatus(mood="tired")


def test_parse_at_place() -> None:
    assert parse_intent("I'm at the gym") == SetStatus(place="the gym")


def test_parse_home_safe() -> None:
    assert parse_intent("I got home safe") == SetStatus(place="home", arrived_safe=True)


def test_parse_status_queries() -> None:
    assert parse_intent("what's Bestie doing") == StatusQuery(name="bestie")
    assert parse_intent("what is Lakshya doing") == StatusQuery(name="lakshya")
    assert parse_intent("where is Bestie") == StatusQuery(name="bestie")
    assert parse_intent("is Bestie awake") == StatusQuery(name="bestie")


def test_parse_non_intent_returns_none() -> None:
    assert parse_intent("tell me a joke") is None
    assert parse_intent("what's 2 plus 2") is None
    assert parse_intent("set a timer for 5 minutes") is None


# --- handling ----------------------------------------------------------------
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


def test_handle_set_status_updates_the_caller() -> None:
    circle = _circle()
    status = StatusService(circle, InMemoryStatusStore())
    reply = handle_intent(circle, status, "u-india", SetStatus(text="coding"), now=NOW)
    assert reply is not None
    assert "coding" in reply.lower()
    mine = status.get_status("u-india", "u-india")
    assert mine is not None and mine.text == "coding"


def test_handle_home_safe_acknowledges_and_records() -> None:
    circle = _circle()
    status = StatusService(circle, InMemoryStatusStore())
    reply = handle_intent(
        circle, status, "u-india", SetStatus(place="home", arrived_safe=True), now=NOW
    )
    assert reply is not None and "safe" in reply.lower()
    mine = status.get_status("u-india", "u-india")
    assert mine is not None and mine.place == "home" and mine.arrived_safe is True


def test_handle_status_query_resolves_a_name() -> None:
    circle = _circle()
    status = StatusService(circle, InMemoryStatusStore())
    status.set_status("u-us", text="having lunch", now=NOW)
    reply = handle_intent(circle, status, "u-india", StatusQuery(name="bestie"), now=NOW)
    assert reply is not None
    assert "Bestie" in reply and "having lunch" in reply


def test_handle_me_resolves_to_the_caller() -> None:
    circle = _circle()
    status = StatusService(circle, InMemoryStatusStore())
    status.set_status("u-india", text="working", now=NOW)
    reply = handle_intent(circle, status, "u-india", StatusQuery(name="me"), now=NOW)
    assert reply is not None and "working" in reply


def test_handle_unknown_name_falls_through() -> None:
    circle = _circle()
    status = StatusService(circle, InMemoryStatusStore())
    # Not someone in the circle -> None so the general assistant handles it.
    assert (
        handle_intent(circle, status, "u-india", StatusQuery(name="the weather"), now=NOW)
        is None
    )
