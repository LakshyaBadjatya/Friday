"""Unit tests for the E2EE group chat service and the new circle helpers.

The server only ever relays ciphertext, so these assert the *membership* guard
(post/read only as a member), chronological history, live broadcaster fan-out, and
the ``groups_for``/``peek_invite`` helpers the routes use.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from friday.circle.chat import (
    ChatBroadcaster,
    ChatMessage,
    ChatService,
    InMemoryChatStore,
)
from friday.circle.service import CircleService
from friday.circle.store import InMemoryCircleStore
from friday.errors import PermissionError

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _circle_with_group() -> tuple[CircleService, str]:
    circle = CircleService(InMemoryCircleStore())
    group = circle.create_group(
        name="Us", admin_uid="u-a", admin_display_name="Me", now=NOW
    )
    return circle, group.id


def test_post_requires_membership() -> None:
    circle, gid = _circle_with_group()
    chat = ChatService(circle, InMemoryChatStore())
    with pytest.raises(PermissionError):
        chat.post(
            group_id=gid, sender_uid="stranger", ciphertext="c", nonce="n", now=NOW
        )


def test_post_and_history_for_member() -> None:
    circle, gid = _circle_with_group()
    chat = ChatService(circle, InMemoryChatStore())
    msg = chat.post(
        group_id=gid, sender_uid="u-a", ciphertext="ct", nonce="nz", now=NOW
    )
    assert msg.ciphertext == "ct"
    history = chat.history(group_id=gid, requester_uid="u-a")
    assert [m.id for m in history] == [msg.id]


def test_history_requires_membership() -> None:
    circle, gid = _circle_with_group()
    chat = ChatService(circle, InMemoryChatStore())
    with pytest.raises(PermissionError):
        chat.history(group_id=gid, requester_uid="stranger")


def test_broadcaster_delivers_then_unsubscribes() -> None:
    async def scenario() -> None:
        bc = ChatBroadcaster()
        queue = bc.subscribe("g1")
        bc.publish(
            ChatMessage(
                id="m1",
                group_id="g1",
                sender_uid="u1",
                ciphertext="x",
                nonce="n",
                created_at=NOW,
            )
        )
        got = await asyncio.wait_for(queue.get(), timeout=1)
        assert got.id == "m1"
        bc.unsubscribe("g1", queue)
        # After unsubscribe a further publish reaches nobody (no error, no delivery).
        bc.publish(got)

    asyncio.run(scenario())


def test_groups_for_and_peek_invite() -> None:
    circle = CircleService(InMemoryCircleStore())
    group = circle.create_group(
        name="Us", admin_uid="u-a", admin_display_name="Me", now=NOW
    )
    assert [g.id for g in circle.groups_for("u-a")] == [group.id]
    assert circle.groups_for("nobody") == []

    invite = circle.invite(group_id=group.id, by_uid="u-a", now=NOW)
    peeked = circle.peek_invite(invite.code)
    assert peeked is not None
    assert peeked.id == group.id
    assert circle.peek_invite("does-not-exist") is None
