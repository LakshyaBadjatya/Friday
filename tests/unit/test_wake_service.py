# © Lakshya Badjatya — Author
"""Unit tests for the wake service (transcript -> WakeEvent + greeting)."""

from __future__ import annotations

from friday.voice.wake_service import WakeService

_OPERATORS = ["FRIDAY", "EDITH", "VISION", "GECKO"]


def test_wake_greets_as_friday() -> None:
    svc = WakeService(_OPERATORS, owner_address="Boss")
    event = svc.handle_transcript("hey friday")
    assert event is not None
    assert event.type == "wake"
    assert event.operator == "FRIDAY"
    assert event.greeting == "I'm up, Boss."


def test_summon_greets_in_that_operators_voice() -> None:
    svc = WakeService(_OPERATORS, owner_address="Boss")
    event = svc.handle_transcript("Friday summon Vision")
    assert event is not None
    assert event.type == "summon"
    assert event.operator == "VISION"
    assert event.greeting == "VISION here, Boss."


def test_owner_address_is_used() -> None:
    svc = WakeService(_OPERATORS, owner_address="Sir")
    assert svc.handle_transcript("hey friday").greeting == "I'm up, Sir."  # type: ignore[union-attr]


def test_non_command_is_none() -> None:
    svc = WakeService(_OPERATORS)
    assert svc.handle_transcript("what's the weather") is None
    assert svc.handle_transcript("") is None
