"""Unit tests for circle messaging: consent-gated DMs, threads, nudges.

You can message someone only while you share a group with them. A thread is both
directions between two people in chronological order; a nudge is a convenience
"thinking of you" message. All offline; the reference instant is passed in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from friday import errors
from friday.circle.messages import InMemoryMessageStore, MessageService
from friday.circle.service import CircleService
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)


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


def _messages(circle: CircleService) -> MessageService:
    return MessageService(circle, InMemoryMessageStore())


def test_send_and_inbox_with_consent() -> None:
    svc = _messages(_circle())
    svc.send(from_uid="u-india", to_uid="u-us", text="call me at 8?", now=NOW)
    inbox = svc.inbox("u-us")
    assert [m.text for m in inbox] == ["call me at 8?"]
    assert svc.unread_count("u-us") == 1


def test_messaging_without_a_shared_group_is_denied() -> None:
    svc = _messages(_circle())
    with pytest.raises(errors.PermissionError):
        svc.send(from_uid="u-india", to_uid="u-stranger", text="hi", now=NOW)


def test_thread_is_both_directions_in_order() -> None:
    svc = _messages(_circle())
    svc.send(from_uid="u-india", to_uid="u-us", text="morning!", now=NOW)
    svc.send(from_uid="u-us", to_uid="u-india", text="hey!", now=NOW + timedelta(minutes=1))
    svc.send(
        from_uid="u-india", to_uid="u-us", text="lunch?", now=NOW + timedelta(minutes=2)
    )
    thread = svc.thread("u-india", "u-us")
    assert [m.text for m in thread] == ["morning!", "hey!", "lunch?"]


def test_mark_read_reduces_unread() -> None:
    svc = _messages(_circle())
    message = svc.send(from_uid="u-india", to_uid="u-us", text="hi", now=NOW)
    assert svc.unread_count("u-us") == 1
    assert svc.mark_read(message.id) is True
    assert svc.unread_count("u-us") == 0


def test_nudge_sends_a_default_message() -> None:
    svc = _messages(_circle())
    nudge = svc.nudge(from_uid="u-india", to_uid="u-us", now=NOW)
    assert nudge.to_uid == "u-us"
    assert nudge.text  # non-empty default
    assert svc.unread_count("u-us") == 1
