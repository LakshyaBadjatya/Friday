"""GitHub, the calendar and the phone — and who gets asked what.

The recurring failure this file guards against is a confident empty answer. An
alert list that is empty because scanning was never switched on looks exactly
like one that is empty because the code is clean, and telling somebody their app
is safe when nothing has ever checked it is worse than saying nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from friday.discord import operators
from friday.link import checks, github


class _Settings:
    def __init__(self, token: str = "tok", repo: str = "") -> None:
        self.github_token = type("S", (), {"get_secret_value": lambda _: token})()
        self.github_default_repo = repo


@pytest.mark.anyio
async def test_scanning_switched_off_is_not_reported_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _forbidden(token: str, path: str) -> tuple[bool, Any]:
        return False, {"status": 403}

    monkeypatch.setattr(github, "_get", _forbidden)
    found = await github.security_alerts(_Settings(), "me/app")
    assert found["ok"] is True
    assert found["scanning_enabled"] is False
    assert found["alerts"] == []
    assert "switched off" in found["note"]


@pytest.mark.anyio
async def test_a_repository_it_cannot_see_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _missing(token: str, path: str) -> tuple[bool, Any]:
        return False, {"status": 404}

    monkeypatch.setattr(github, "_get", _missing)
    found = await github.build_status(_Settings(), "me/nope")
    assert found["ok"] is False
    assert "check the name" in found["error"]


@pytest.mark.anyio
async def test_real_runs_come_back_with_what_failed_and_where(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _runs(token: str, path: str) -> tuple[bool, Any]:
        return True, {"workflow_runs": [{
            "name": "tests", "status": "completed", "conclusion": "failure",
            "head_branch": "main", "updated_at": "2026-08-16T10:00:00Z",
            "head_commit": {"message": "tighten the port filter"},
        }]}

    monkeypatch.setattr(github, "_get", _runs)
    found = await github.build_status(_Settings(), "me/app")
    assert found["ok"] is True
    assert found["runs"][0]["conclusion"] == "failure"
    assert found["runs"][0]["branch"] == "main"


@pytest.mark.anyio
async def test_no_token_is_a_stated_gap_not_an_empty_list() -> None:
    empty = _Settings(token="")
    assert (await github.build_status(empty, "me/app"))["ok"] is False
    assert await github.repositories(empty) == []


@pytest.mark.anyio
async def test_no_google_account_means_say_so_rather_than_guess() -> None:
    class App:
        state = type("S", (), {"settings": type("T", (), {
            "google_oauth_token": None})()})()

    told = await operators.calendar_report(App(), "what's on tomorrow")
    assert told is not None
    assert "No Google account is connected" in told
    assert "do not guess" in told


@pytest.mark.parametrize(
    ("text", "calendar", "repo"),
    [
        ("what's on my calendar tomorrow", True, False),
        ("what do i have today", True, False),
        ("remind me at 6 to call mum", False, False),
        ("is my android app safe", False, True),
        ("check lakshya/friday for vulnerabilities", False, True),
        ("check the security of my devices", False, False),
    ],
)
def test_the_right_source_is_consulted(text: str, calendar: bool, repo: bool) -> None:
    """A repository question is about code; everything else is about the machine."""
    assert operators.wants_calendar(text) is calendar
    assert operators.mentions_repo(text) is repo


def test_a_phone_and_a_laptop_are_scanned_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Termux reports itself through the environment, not through platform."""
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert checks.on_android() is True
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(checks.Path, "exists", lambda self: False)
    assert checks.on_android() is False
