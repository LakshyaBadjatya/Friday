"""Guardrail layer over a :class:`~friday.circle.store.CircleStore`.

All the rules live here (the store is dumb persistence): only an admin may invite,
revoke, or remove; an invite is single-use and time-boxed; and ``shares_group`` is
the consent primitive every downstream feature checks before revealing anything
about a person — two people see each other only while they share a group.

The reference instant (``now``) is always passed in so invite expiry is
deterministic and testable.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from uuid import uuid4

from friday.circle.models import Group, Invite, InviteError, Member, Role
from friday.circle.store import CircleStore
from friday.errors import PermissionError

#: Default lifetime of an invite code.
_DEFAULT_TTL = timedelta(days=7)
#: Bytes of entropy for a generated invite code (~12 url-safe chars).
_CODE_BYTES = 9


class CircleService:
    """Group/membership operations with role + consent guardrails."""

    def __init__(self, store: CircleStore) -> None:
        self._store = store

    def create_group(
        self,
        *,
        name: str,
        admin_uid: str,
        admin_display_name: str,
        admin_tz: str = "UTC",
        description: str = "",
        now: datetime,
        group_id: str | None = None,
    ) -> Group:
        """Create a group and seed its creator as the first ADMIN member."""
        gid = group_id or uuid4().hex
        group = Group(
            id=gid,
            name=name,
            description=description,
            admin_uid=admin_uid,
            created_at=now,
        )
        self._store.create_group(group)
        self._store.add_member(
            gid,
            Member(
                uid=admin_uid,
                display_name=admin_display_name,
                role=Role.ADMIN,
                tz=admin_tz,
                joined_at=now,
            ),
        )
        return group

    def list_members(self, group_id: str) -> list[Member]:
        return self._store.list_members(group_id)

    def invite(
        self,
        *,
        group_id: str,
        by_uid: str,
        email: str | None = None,
        now: datetime,
        ttl: timedelta = _DEFAULT_TTL,
        code: str | None = None,
    ) -> Invite:
        """Mint a single-use invite; only an admin of the group may do this."""
        self._require_admin(group_id, by_uid)
        invite = Invite(
            code=code or secrets.token_urlsafe(_CODE_BYTES),
            group_id=group_id,
            created_by=by_uid,
            email=email,
            created_at=now,
            expires_at=now + ttl,
        )
        self._store.save_invite(invite)
        return invite

    def accept_invite(
        self,
        *,
        code: str,
        uid: str,
        display_name: str,
        tz: str = "UTC",
        now: datetime,
    ) -> Member:
        """Join a group via an invite code, enforcing validity guardrails."""
        invite = self._store.get_invite(code)
        if invite is None:
            raise InviteError("invite not found")
        if invite.revoked:
            raise InviteError("invite revoked")
        if invite.accepted_by is not None:
            raise InviteError("invite already used")
        if now >= invite.expires_at:
            raise InviteError("invite expired")
        member = Member(
            uid=uid,
            display_name=display_name,
            role=Role.MEMBER,
            tz=tz,
            joined_at=now,
        )
        self._store.add_member(invite.group_id, member)
        self._store.save_invite(invite.model_copy(update={"accepted_by": uid}))
        return member

    def revoke_invite(self, *, code: str, by_uid: str) -> None:
        """Revoke an outstanding invite; only an admin of its group may do this."""
        invite = self._store.get_invite(code)
        if invite is None:
            raise InviteError("invite not found")
        self._require_admin(invite.group_id, by_uid)
        self._store.save_invite(invite.model_copy(update={"revoked": True}))

    def remove_member(self, *, group_id: str, uid: str, by_uid: str) -> bool:
        """Remove a member; only an admin may do this. Returns whether one was removed."""
        self._require_admin(group_id, by_uid)
        return self._store.remove_member(group_id, uid)

    def shares_group(self, uid_a: str, uid_b: str) -> bool:
        """The consent primitive: do two people currently share any group?"""
        return bool(self._store.groups_of(uid_a) & self._store.groups_of(uid_b))

    def find_member(self, uid: str) -> Member | None:
        """Return any membership record for ``uid`` (for its display_name/tz).

        A person carries the same identity across the groups they're in, so the
        first membership found is sufficient for naming/timezone.
        """
        for group_id in self._store.groups_of(uid):
            member = self._store.get_member(group_id, uid)
            if member is not None:
                return member
        return None

    def members_visible_to(self, uid: str) -> list[Member]:
        """Every distinct member across the groups ``uid`` belongs to (incl. self)."""
        seen: dict[str, Member] = {}
        for group_id in self._store.groups_of(uid):
            for member in self._store.list_members(group_id):
                seen.setdefault(member.uid, member)
        return list(seen.values())

    def _require_admin(self, group_id: str, uid: str) -> None:
        member = self._store.get_member(group_id, uid)
        if member is None or member.role is not Role.ADMIN:
            raise PermissionError(f"{uid!r} is not an admin of {group_id!r}")
