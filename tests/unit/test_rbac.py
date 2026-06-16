# © Lakshya Badjatya — Author
"""Unit tests for role-scoped access control (default-deny)."""

from __future__ import annotations

import pytest

from friday.security.rbac import AccessPolicy, Role

_OWNER = Role(name="owner", permissions=frozenset({"*"}))
_MEMBER = Role(name="member", permissions=frozenset({"reminders.read", "reminders.add"}))


def test_wildcard_role_can_do_anything() -> None:
    policy = AccessPolicy([_OWNER, _MEMBER], {"boss": "owner"})
    assert policy.can("boss", "reminders.add") is True
    assert policy.can("boss", "anything.at.all") is True


def test_member_limited_to_granted_permissions() -> None:
    policy = AccessPolicy([_OWNER, _MEMBER], {"kid": "member"})
    assert policy.can("kid", "reminders.read") is True
    assert policy.can("kid", "reminders.delete") is False  # not granted


def test_unassigned_user_denied() -> None:
    policy = AccessPolicy([_OWNER, _MEMBER])
    assert policy.can("stranger", "reminders.read") is False
    assert policy.role_of("stranger") is None


def test_assign_unknown_role_rejected() -> None:
    policy = AccessPolicy([_OWNER])
    with pytest.raises(ValueError, match="unknown role"):
        policy.assign("x", "ghost")
    with pytest.raises(ValueError, match="unknown role"):
        AccessPolicy([_OWNER], {"x": "ghost"})


def test_assign_then_check() -> None:
    policy = AccessPolicy([_OWNER, _MEMBER])
    policy.assign("kid", "member")
    assert policy.role_of("kid") == "member"
    assert policy.can("kid", "reminders.add") is True
