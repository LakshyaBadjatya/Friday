"""Storage interface for the circle, plus an in-memory implementation.

The store is dumb persistence (no guardrails — those live in
:class:`~friday.circle.service.CircleService`). :class:`InMemoryCircleStore` backs
the offline tests and local runs; a Firestore-backed implementation of the same
:class:`CircleStore` protocol is wired in later without touching callers.
"""

from __future__ import annotations

from typing import Protocol

from friday.circle.models import Group, Invite, Member


class CircleStore(Protocol):
    """The persistence surface the service depends on (storage-agnostic)."""

    def create_group(self, group: Group) -> None: ...

    def get_group(self, group_id: str) -> Group | None: ...

    def add_member(self, group_id: str, member: Member) -> None: ...

    def get_member(self, group_id: str, uid: str) -> Member | None: ...

    def list_members(self, group_id: str) -> list[Member]: ...

    def remove_member(self, group_id: str, uid: str) -> bool: ...

    def groups_of(self, uid: str) -> set[str]: ...

    def save_invite(self, invite: Invite) -> None: ...

    def get_invite(self, code: str) -> Invite | None: ...


class InMemoryCircleStore:
    """A dict-backed :class:`CircleStore` for tests and local use."""

    def __init__(self) -> None:
        self._groups: dict[str, Group] = {}
        # group_id -> (uid -> Member), insertion-ordered.
        self._members: dict[str, dict[str, Member]] = {}
        self._invites: dict[str, Invite] = {}

    def create_group(self, group: Group) -> None:
        self._groups[group.id] = group
        self._members.setdefault(group.id, {})

    def get_group(self, group_id: str) -> Group | None:
        return self._groups.get(group_id)

    def add_member(self, group_id: str, member: Member) -> None:
        self._members.setdefault(group_id, {})[member.uid] = member

    def get_member(self, group_id: str, uid: str) -> Member | None:
        return self._members.get(group_id, {}).get(uid)

    def list_members(self, group_id: str) -> list[Member]:
        return list(self._members.get(group_id, {}).values())

    def remove_member(self, group_id: str, uid: str) -> bool:
        return self._members.get(group_id, {}).pop(uid, None) is not None

    def groups_of(self, uid: str) -> set[str]:
        return {gid for gid, members in self._members.items() if uid in members}

    def save_invite(self, invite: Invite) -> None:
        self._invites[invite.code] = invite

    def get_invite(self, code: str) -> Invite | None:
        return self._invites.get(code)
