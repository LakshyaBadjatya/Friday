"""Domain models for the circle: groups, members, and invites.

Pydantic models (mirroring the family store's ``Participant``/``ViewRecord``) so
they round-trip cleanly to whatever backing store holds them. Identity (``uid``)
comes from the auth layer; nothing here is personal or persisted at import time.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from friday.errors import FridayError


class InviteError(FridayError):
    """An invite could not be accepted (unknown, expired, used, or revoked)."""


class Role(StrEnum):
    """A member's role within a group."""

    ADMIN = "admin"
    MEMBER = "member"


class Member(BaseModel):
    """One person's membership of a group."""

    uid: str = Field(min_length=1)
    display_name: str = Field(min_length=1, max_length=200)
    role: Role
    #: IANA timezone (e.g. ``"Asia/Kolkata"``) for the timezone-aware features.
    tz: str = "UTC"
    joined_at: datetime


class Group(BaseModel):
    """A circle: a named group with one creating admin."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    admin_uid: str = Field(min_length=1)
    created_at: datetime


class Invite(BaseModel):
    """A single-use invitation into a group (by code, optionally tied to email)."""

    code: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    email: str | None = None
    created_at: datetime
    expires_at: datetime
    accepted_by: str | None = None
    revoked: bool = False
