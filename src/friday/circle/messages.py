"""Messaging for the circle: consent-gated DMs, threads, and "thinking of you" nudges.

You can message someone only while you share a group with them
(:meth:`CircleService.shares_group`). A thread is both directions between two
people, chronological. A nudge is a one-tap message with a warm default. Storage
is behind :class:`MessageStore`; the in-memory implementation backs tests and
local runs (a persistent implementation is wired later).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel

from friday.circle.service import CircleService
from friday.errors import PermissionError

#: The default text a nudge sends.
_NUDGE_TEXT = "Thinking of you. 💭"


class Message(BaseModel):
    """One direct message between two people."""

    id: str
    from_uid: str
    to_uid: str
    text: str
    created_at: datetime
    read: bool = False


class MessageStore(Protocol):
    """Persistence for messages (insertion order is preserved)."""

    def save(self, message: Message) -> None: ...

    def get(self, message_id: str) -> Message | None: ...

    def list_all(self) -> list[Message]: ...


class InMemoryMessageStore:
    """A dict-backed :class:`MessageStore` for tests and local use."""

    def __init__(self) -> None:
        self._by_id: dict[str, Message] = {}

    def save(self, message: Message) -> None:
        self._by_id[message.id] = message

    def get(self, message_id: str) -> Message | None:
        return self._by_id.get(message_id)

    def list_all(self) -> list[Message]:
        return list(self._by_id.values())


class MessageService:
    """Send/read messages between circle members, enforcing the consent guardrail."""

    def __init__(self, circle: CircleService, store: MessageStore) -> None:
        self._circle = circle
        self._store = store

    def send(
        self,
        *,
        from_uid: str,
        to_uid: str,
        text: str,
        now: datetime,
        message_id: str | None = None,
    ) -> Message:
        """Send a message; requires the two to share a group."""
        if not self._circle.shares_group(from_uid, to_uid):
            raise PermissionError(
                f"{from_uid!r} cannot message {to_uid!r} (no shared group)"
            )
        message = Message(
            id=message_id or uuid4().hex,
            from_uid=from_uid,
            to_uid=to_uid,
            text=text,
            created_at=now,
        )
        self._store.save(message)
        return message

    def nudge(self, *, from_uid: str, to_uid: str, now: datetime) -> Message:
        """Send a one-tap "thinking of you" message."""
        return self.send(from_uid=from_uid, to_uid=to_uid, text=_NUDGE_TEXT, now=now)

    def inbox(self, uid: str) -> list[Message]:
        """Messages addressed to ``uid``, chronological."""
        return [m for m in self._store.list_all() if m.to_uid == uid]

    def unread_count(self, uid: str) -> int:
        return sum(1 for m in self._store.list_all() if m.to_uid == uid and not m.read)

    def thread(self, uid_a: str, uid_b: str) -> list[Message]:
        """Both directions between two people, chronological (consent-gated)."""
        if not self._circle.shares_group(uid_a, uid_b):
            raise PermissionError(f"{uid_a!r} and {uid_b!r} do not share a group")
        pair = {uid_a, uid_b}
        return [m for m in self._store.list_all() if {m.from_uid, m.to_uid} == pair]

    def mark_read(self, message_id: str) -> bool:
        message = self._store.get(message_id)
        if message is None or message.read:
            return False
        self._store.save(message.model_copy(update={"read": True}))
        return True
