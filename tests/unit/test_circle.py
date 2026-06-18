"""Unit tests for the circle data model: store + service guardrails.

Fully offline against the in-memory store (the Firestore adapter is wired later
behind the same interface). The reference instant is passed in everywhere so the
invite-expiry logic is deterministic with no clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from friday import errors
from friday.circle.models import InviteError, Role
from friday.circle.service import CircleService
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)


def _service() -> CircleService:
    return CircleService(InMemoryCircleStore())


def _group_with_admin(svc: CircleService) -> str:
    group = svc.create_group(
        name="Inner Circle",
        admin_uid="u-admin",
        admin_display_name="Admin",
        admin_tz="Asia/Kolkata",
        now=NOW,
        group_id="g1",
    )
    return group.id


def test_create_group_makes_creator_an_admin_member() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    members = svc.list_members(gid)
    assert len(members) == 1
    assert members[0].uid == "u-admin"
    assert members[0].role is Role.ADMIN
    assert members[0].tz == "Asia/Kolkata"


def test_admin_can_invite_and_member_can_accept() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    invite = svc.invite(group_id=gid, by_uid="u-admin", email="friend@x.test", now=NOW)
    member = svc.accept_invite(
        code=invite.code,
        uid="u-friend",
        display_name="Friend",
        tz="America/New_York",
        now=NOW + timedelta(minutes=1),
    )
    assert member.role is Role.MEMBER
    assert member.tz == "America/New_York"
    assert {m.uid for m in svc.list_members(gid)} == {"u-admin", "u-friend"}


def test_non_admin_cannot_invite() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    svc.accept_invite(
        code=svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code,
        uid="u-friend",
        display_name="Friend",
        now=NOW,
    )
    with pytest.raises(errors.PermissionError):
        svc.invite(group_id=gid, by_uid="u-friend", now=NOW)


def test_invite_cannot_be_reused() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    code = svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code
    svc.accept_invite(code=code, uid="u-a", display_name="A", now=NOW)
    with pytest.raises(InviteError):
        svc.accept_invite(code=code, uid="u-b", display_name="B", now=NOW)


def test_expired_invite_is_rejected() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    code = svc.invite(
        group_id=gid, by_uid="u-admin", now=NOW, ttl=timedelta(days=7)
    ).code
    with pytest.raises(InviteError):
        svc.accept_invite(
            code=code, uid="u-late", display_name="Late", now=NOW + timedelta(days=8)
        )


def test_revoked_invite_is_rejected_and_only_admin_revokes() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    code = svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code
    with pytest.raises(errors.PermissionError):
        svc.revoke_invite(code=code, by_uid="u-stranger")
    svc.revoke_invite(code=code, by_uid="u-admin")
    with pytest.raises(InviteError):
        svc.accept_invite(code=code, uid="u-friend", display_name="Friend", now=NOW)


def test_unknown_invite_is_rejected() -> None:
    svc = _service()
    with pytest.raises(InviteError):
        svc.accept_invite(code="nope", uid="u", display_name="U", now=NOW)


def test_shares_group_is_the_consent_primitive() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    svc.accept_invite(
        code=svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code,
        uid="u-friend",
        display_name="Friend",
        now=NOW,
    )
    assert svc.shares_group("u-admin", "u-friend") is True
    assert svc.shares_group("u-admin", "u-stranger") is False


def test_removing_a_member_drops_the_shared_consent() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    svc.accept_invite(
        code=svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code,
        uid="u-friend",
        display_name="Friend",
        now=NOW,
    )
    assert svc.remove_member(group_id=gid, uid="u-friend", by_uid="u-admin") is True
    assert svc.shares_group("u-admin", "u-friend") is False


def test_only_admin_can_remove_members() -> None:
    svc = _service()
    gid = _group_with_admin(svc)
    svc.accept_invite(
        code=svc.invite(group_id=gid, by_uid="u-admin", now=NOW).code,
        uid="u-friend",
        display_name="Friend",
        now=NOW,
    )
    with pytest.raises(errors.PermissionError):
        svc.remove_member(group_id=gid, uid="u-admin", by_uid="u-friend")
