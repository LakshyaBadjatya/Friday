"""End-to-end-encrypted group chat for the circle.

The server is a blind relay: a message arrives already encrypted (AES-GCM, done in
the browser with a key the backend never sees) and only ciphertext + nonce are
stored. The single guardrail is membership — you may post to, or read, a group only
while you are a member of it (mirroring
:meth:`~friday.circle.service.CircleService.shares_group`).

:class:`ChatBroadcaster` is an in-process fan-out so a Server-Sent-Events endpoint
can push new ciphertext to connected members live. It is per-process (one Render
instance), which suits a small private circle; history always comes from the store,
so a missed broadcast is recovered on the next fetch.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from friday.circle.service import CircleService
from friday.errors import PermissionError


class ChatMessage(BaseModel):
    """One end-to-end-encrypted group message (server sees only ciphertext)."""

    id: str
    group_id: str = Field(min_length=1)
    sender_uid: str = Field(min_length=1)
    #: Base64 AES-GCM ciphertext and the 96-bit nonce it was sealed with.
    ciphertext: str = Field(min_length=1, max_length=20000)
    nonce: str = Field(min_length=1, max_length=64)
    created_at: datetime


class ChatStore(Protocol):
    """Persistence for chat messages (chronological within a group)."""

    def save(self, message: ChatMessage) -> None: ...

    def history(self, group_id: str, limit: int = 200) -> list[ChatMessage]: ...


class InMemoryChatStore:
    """A list-backed :class:`ChatStore` for tests and local use."""

    def __init__(self) -> None:
        self._by_group: dict[str, list[ChatMessage]] = {}

    def save(self, message: ChatMessage) -> None:
        self._by_group.setdefault(message.group_id, []).append(message)

    def history(self, group_id: str, limit: int = 200) -> list[ChatMessage]:
        msgs = self._by_group.get(group_id, [])
        return msgs[-limit:]


class ChatBroadcaster:
    """In-process pub/sub: fan a posted message out to live SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[ChatMessage]]] = {}

    def subscribe(self, group_id: str) -> asyncio.Queue[ChatMessage]:
        queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self._subscribers.setdefault(group_id, set()).add(queue)
        return queue

    def unsubscribe(self, group_id: str, queue: asyncio.Queue[ChatMessage]) -> None:
        subs = self._subscribers.get(group_id)
        if subs is not None:
            subs.discard(queue)
            if not subs:
                self._subscribers.pop(group_id, None)

    def publish(self, message: ChatMessage) -> None:
        for queue in self._subscribers.get(message.group_id, set()):
            queue.put_nowait(message)


class StreamTicket(BaseModel):
    """A short-lived authorization to open one SSE stream (uid bound to a group)."""

    uid: str
    group_id: str
    expires_at: datetime


class StreamTicketStore:
    """Mint single-use, short-TTL tickets so the SSE URL never carries a token.

    ``EventSource`` can't send an ``Authorization`` header, and putting the bearer
    token in the query string would leak it into access logs and browser history.
    Instead the client authenticates a normal POST (header), gets an opaque ticket
    bound to ``(uid, group_id)``, and opens the stream with that. The ticket is
    popped on first use and expires quickly, so a leaked one is inert.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._ttl = ttl_seconds
        self._tickets: dict[str, StreamTicket] = {}

    def mint(self, *, uid: str, group_id: str, now: datetime) -> str:
        token = secrets.token_urlsafe(24)
        self._tickets[token] = StreamTicket(
            uid=uid,
            group_id=group_id,
            expires_at=now + timedelta(seconds=self._ttl),
        )
        return token

    def consume(self, ticket: str, *, now: datetime) -> StreamTicket | None:
        """Atomically take a ticket; ``None`` if unknown or expired (single-use)."""
        entry = self._tickets.pop(ticket, None)
        if entry is None or now >= entry.expires_at:
            return None
        return entry


class ChatService:
    """Post/read E2EE group messages, gated by current group membership."""

    def __init__(
        self,
        circle: CircleService,
        store: ChatStore,
        broadcaster: ChatBroadcaster | None = None,
    ) -> None:
        self._circle = circle
        self._store = store
        self._broadcaster = broadcaster or ChatBroadcaster()

    @property
    def broadcaster(self) -> ChatBroadcaster:
        return self._broadcaster

    def _require_member(self, group_id: str, uid: str) -> None:
        if not any(m.uid == uid for m in self._circle.list_members(group_id)):
            raise PermissionError(f"{uid!r} is not a member of {group_id!r}")

    def post(
        self,
        *,
        group_id: str,
        sender_uid: str,
        ciphertext: str,
        nonce: str,
        now: datetime,
        message_id: str | None = None,
    ) -> ChatMessage:
        """Store an already-encrypted message and fan it out to subscribers."""
        self._require_member(group_id, sender_uid)
        message = ChatMessage(
            id=message_id or uuid4().hex,
            group_id=group_id,
            sender_uid=sender_uid,
            ciphertext=ciphertext,
            nonce=nonce,
            created_at=now,
        )
        self._store.save(message)
        self._broadcaster.publish(message)
        return message

    def history(
        self, *, group_id: str, requester_uid: str, limit: int = 200
    ) -> list[ChatMessage]:
        """Chronological ciphertext history for a member of the group."""
        self._require_member(group_id, requester_uid)
        return self._store.history(group_id, limit)
