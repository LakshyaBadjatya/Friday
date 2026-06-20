"""Unit tests for the Firestore-linked Siri circle handler.

A fake :class:`FirestoreRest` (in-memory) + a stubbed ``resolve_token`` keep these
offline and deterministic. They assert the spoken reply for each A–C intent and
that an unknown name falls through (``None``) to the general assistant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import friday.circle.siri_circle as sc

# Noon UTC → 5:30 PM Kolkata (caller) / 7:00 AM New York (friend, awake).
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class FakeFs:
    """In-memory stand-in for FirestoreRest (records writes, serves canned reads)."""

    created: list[tuple[str, dict[str, Any]]] = []
    patched: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, _id_token: str) -> None:
        self._members = [
            {"uid": "u-a", "displayName": "Me", "tz": "Asia/Kolkata", "presence": "active"},
            {
                "uid": "u-b",
                "displayName": "Bestie",
                "tz": "America/New_York",
                "presence": "away",
            },
        ]

    def list(self, path: str) -> list[dict[str, Any]]:
        if path == "users/u-a/memberships":
            return [{"groupId": "g1", "groupName": "Us"}]
        if path == "groups/g1/members":
            return self._members
        return []

    def get(self, path: str) -> dict[str, Any] | None:
        if path == "groups/g1":
            return {"name": "Us", "adminUid": "u-a"}
        if path.startswith("groups/g1/members/"):
            uid = path.rsplit("/", 1)[-1]
            return next((m for m in self._members if m["uid"] == uid), None)
        return None

    def patch(self, path: str, fields: dict[str, Any]) -> bool:
        FakeFs.patched.append((path, fields))
        return True

    def create(self, path: str, fields: dict[str, Any]) -> bool:
        FakeFs.created.append((path, fields))
        return True


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFs.created = []
    FakeFs.patched = []
    monkeypatch.setattr(sc, "resolve_token", lambda _t: ("id-token", "u-a"))
    monkeypatch.setattr(sc, "FirestoreRest", FakeFs)


def _ask(query: str) -> str | None:
    return sc.handle("a-long-refresh-token", query, NOW)


def test_status_query_speaks_friends_local_time() -> None:
    reply = _ask("what's Bestie doing")
    assert reply is not None
    assert "Bestie" in reply and ("AM" in reply or "PM" in reply)


def test_good_time_to_call_resolves_friend() -> None:
    reply = _ask("is it a good time to call Bestie")
    assert reply is not None
    assert "Bestie" in reply


def test_set_status_patches_self() -> None:
    reply = _ask("set my status to heads down")
    assert reply is not None and "status set to heads down" in reply.lower()
    assert any("members/u-a" in p for p, _ in FakeFs.patched)


def test_safe_arrival() -> None:
    reply = _ask("tell friday i'm home safe")
    assert reply is not None and "home safe" in reply.lower()


def test_nudge_creates_metadata() -> None:
    reply = _ask("send Bestie a thinking of you")
    assert reply is not None and "thinking of them" in reply.lower()
    assert any(p.endswith("/nudges") for p, _ in FakeFs.created)


def test_sos_alerts_circle() -> None:
    reply = _ask("SOS")
    assert reply is not None and "sos sent" in reply.lower()
    assert any(p.endswith("/alerts") for p, _ in FakeFs.created)


def test_reminder_for_friend() -> None:
    reply = _ask("remind Bestie to take her meds at 9pm")
    assert reply is not None and "remind bestie to take her meds" in reply.lower()
    assert any(p.endswith("/reminders") for p, _ in FakeFs.created)


def test_weather_here_uses_wttr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "_wttr", lambda _loc: "Sunny +31°C")
    reply = _ask("what's the weather")
    assert reply is not None and "31" in reply


def test_weather_with_city_falls_through() -> None:
    # "weather in kota" has an explicit place → orchestrator handles it, not us.
    assert _ask("what's the weather in kota") is None


def test_unknown_name_falls_through() -> None:
    # "the news" isn't a circle member → None so the orchestrator answers.
    assert _ask("what's the news doing") is None
