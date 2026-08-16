"""Who gets summoned, and — mostly — who doesn't.

Half the roster is also an ordinary English word. "forge ahead with that" and
"vision is blurry" are sentences about something else that happen to open with
an operator's name, and a bot that answered them in character would be worse
company than one that stayed quiet. The false-positive cases below are the
point of this file.
"""

from __future__ import annotations

import pytest

from friday.discord import operators


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EDITH what is the weather", "EDITH"),
        ("edith, status?", "EDITH"),
        ("E.D.I.T.H hello", "EDITH"),          # written out, as in the films
        ("hey oracle when is my next reminder", "ORACLE"),
        ("gecko should i buy nvidia", "GECKO"),
        ("jocasta what did i say about the exam", "JOCASTA"),
    ],
)
def test_an_operator_answers_to_its_name(text: str, expected: str) -> None:
    found = operators.addressed(text)
    assert found is not None
    assert found.name == expected


@pytest.mark.parametrize(
    "text",
    [
        "forge ahead with that",       # a verb, not a summons
        "forge is down",
        "vision is blurry",
        "karen and i talked",
        "friday hi",                   # she has her own path
        "hello everyone",
        "lol",
    ],
)
def test_ordinary_english_is_not_a_summons(text: str) -> None:
    assert operators.addressed(text) is None


def test_the_persona_names_itself_and_keeps_the_shared_memory() -> None:
    rule = operators.persona_rule(operators.addressed("gecko should i buy nvidia"))
    assert "You are GECKO, not FRIDAY" in rule
    assert "share FRIDAY's memory" in rule
    # The speciality is an angle, not a gate: an operator asked something off
    # its patch should still answer rather than forward a ticket.
    assert "still answer it" in rule


@pytest.mark.anyio
async def test_no_webhook_permission_falls_back_rather_than_going_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty id tells the caller to answer as FRIDAY instead."""

    async def _denied(*args: object, **kwargs: object) -> tuple[bool, dict]:
        return False, {"code": 50013}  # Missing Permissions

    monkeypatch.setattr(operators, "_call", _denied)
    monkeypatch.setattr(operators, "_CACHE", {})
    edith = operators.addressed("edith, status?")
    assert await operators.speak("tok", "chan", edith, "all quiet") == ""


@pytest.mark.anyio
async def test_an_operator_cannot_be_talked_into_pinging_everyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Webhook posts honour @everyone by default; this one must not."""
    sent: dict[str, object] = {}

    async def _capture(
        method: str, path: str, token: str, body: object, **kwargs: object
    ) -> tuple[bool, dict]:
        if method == "POST" and path.startswith("/webhooks/"):
            sent.update(body or {})  # type: ignore[arg-type]
            return True, {"id": "999"}
        return True, {"id": "1", "token": "hooktok", "name": "FRIDAY Roster"}

    monkeypatch.setattr(operators, "_call", _capture)
    monkeypatch.setattr(operators, "_CACHE", {})
    edith = operators.addressed("edith, status?")
    assert await operators.speak("tok", "chan", edith, "@everyone hi") == "999"
    assert sent["username"] == "EDITH"
    assert sent["allowed_mentions"] == {"parse": ["users"]}
