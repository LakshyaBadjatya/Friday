"""Per-person status: set it, read it (consent-gated), describe it for speech.

A person has one current status (free-form text, mood, coarse place, safe-arrival
flag). Visibility is gated by :meth:`CircleService.shares_group` — you can read
someone's status only while you share a group with them. :meth:`StatusService.describe`
composes a spoken line using the target's timezone, so a reply reads naturally
across a long distance ("… It's 1:00 PM for them. Set 20 minutes ago.").

Storage is behind :class:`StatusStore`; the in-memory implementation backs tests
and local runs, with a persistent implementation wired later.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel

from friday.circle.friend_time import describe_local
from friday.circle.service import CircleService
from friday.errors import PermissionError


class MemberStatus(BaseModel):
    """A person's current shared status (all activity fields optional)."""

    uid: str
    text: str | None = None
    mood: str | None = None
    #: Coarse place label ("home", "gym") — never raw coordinates here.
    place: str | None = None
    arrived_safe: bool | None = None
    updated_at: datetime


class StatusStore(Protocol):
    """Persistence for one status per uid."""

    def get(self, uid: str) -> MemberStatus | None: ...

    def set(self, status: MemberStatus) -> None: ...


class InMemoryStatusStore:
    """A dict-backed :class:`StatusStore` for tests and local use."""

    def __init__(self) -> None:
        self._by_uid: dict[str, MemberStatus] = {}

    def get(self, uid: str) -> MemberStatus | None:
        return self._by_uid.get(uid)

    def set(self, status: MemberStatus) -> None:
        self._by_uid[status.uid] = status


def humanize_ago(delta: timedelta) -> str:
    """Render an elapsed duration as a spoken phrase ('20 minutes ago')."""
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


class StatusService:
    """Set/read/describe a person's status, enforcing the consent guardrail."""

    def __init__(self, circle: CircleService, store: StatusStore) -> None:
        self._circle = circle
        self._store = store

    def set_status(
        self,
        uid: str,
        *,
        text: str | None = None,
        mood: str | None = None,
        place: str | None = None,
        arrived_safe: bool | None = None,
        now: datetime,
    ) -> MemberStatus:
        """Upsert ``uid``'s status, merging only the fields provided this call."""
        existing = self._store.get(uid)
        merged = MemberStatus(
            uid=uid,
            text=text if text is not None else (existing.text if existing else None),
            mood=mood if mood is not None else (existing.mood if existing else None),
            place=place if place is not None else (existing.place if existing else None),
            arrived_safe=(
                arrived_safe
                if arrived_safe is not None
                else (existing.arrived_safe if existing else None)
            ),
            updated_at=now,
        )
        self._store.set(merged)
        return merged

    def get_status(self, viewer_uid: str, target_uid: str) -> MemberStatus | None:
        """Return ``target_uid``'s status, or raise if the viewer has no consent."""
        if not self._circle.shares_group(viewer_uid, target_uid):
            raise PermissionError(
                f"{viewer_uid!r} does not share a group with {target_uid!r}"
            )
        return self._store.get(target_uid)

    def describe(self, viewer_uid: str, target_uid: str, *, now: datetime) -> str:
        """Compose a spoken 'what are they doing' line (consent-gated)."""
        status = self.get_status(viewer_uid, target_uid)
        member = self._circle.find_member(target_uid)
        name = member.display_name if member else target_uid
        tz = member.tz if member else "UTC"

        if status is None:
            return f"{name} hasn't set a status yet."

        view = describe_local(tz, now)
        asleep = " — they're probably asleep" if view.asleep else ""
        ago = humanize_ago(now - status.updated_at)
        return (
            f"{name} is {self._activity_phrase(status)}. "
            f"It's {view.local_time} for them{asleep}. Set {ago}."
        )

    @staticmethod
    def _activity_phrase(status: MemberStatus) -> str:
        if status.text:
            base = status.text
            if status.mood:
                base = f"{base} and feeling {status.mood}"
        elif status.place:
            base = f"at {status.place}"
        elif status.mood:
            base = f"feeling {status.mood}"
        else:
            base = "around"
        if status.arrived_safe:
            base = f"{base} (got there safe)"
        return base
