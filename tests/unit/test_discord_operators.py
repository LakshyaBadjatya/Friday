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


def test_the_persona_rule_does_not_trip_the_identity_guard() -> None:
    """The rules we append must not read as a question the guard answers.

    "edith how are you" came back as the canned "who are you" reply because the
    persona rule contained the words "introduce yourself" and the guard was
    scanning the fully assembled prompt. The real fix is that intent is now read
    from the human's words — this pins the second half, that the rule text is
    not itself question-shaped, so the next detector added upstream does not
    inherit the same trap.
    """
    from friday.siri import guard  # noqa: PLC0415

    for name in ("EDITH", "ORACLE", "GECKO", "VISION"):
        rule = operators.persona_rule(operators.addressed(f"{name.lower()} hi"))
        assert guard._IDENTITY.search(rule) is None, name
        assert guard.blocked(rule) is None, name


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("solve this friday", True),
        ("solve this", True),
        ("friday solve this", True),
        ("can you answer that", True),
        ("explain this properly", True),
        ("what is the capital of france", False),   # carries its own subject
        ("solve x^2 + 2x = 0", False),
        ("hello", False),
    ],
)
def test_an_instruction_with_a_pronoun_means_the_quoted_message(
    text: str, expected: bool
) -> None:
    """Replying "solve this friday" to a problem must solve the problem.

    Those three words were being sent on alone, and she answered — correctly and
    uselessly — that the problem statement appeared to be missing.
    """
    from friday.discord.gateway import _points_at_quote  # noqa: PLC0415

    assert _points_at_quote(text) is expected


def test_a_long_answer_is_split_rather_than_truncated() -> None:
    """Nothing may be lost to Discord's 2000-character limit.

    Trimming to fit removed the substitution check at the end of every worked
    solution — the step that makes the answer worth trusting — and left an
    ellipsis in its place.
    """
    from friday.discord.gateway import split_for_discord  # noqa: PLC0415

    body = "\n".join(f"step {n}: " + "x" * 60 for n in range(60))
    parts = split_for_discord(body)
    assert len(parts) > 1
    assert all(len(part) <= 1900 for part in parts)
    assert "\n".join(parts) == body          # lossless, seams on line breaks

    assert split_for_discord("short") == ["short"]
    assert split_for_discord("   ") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("friday can you check security of my devices please", "EDITH"),
        ("remind me at 6pm to call mum", "ORACLE"),
        ("should i buy nvidia stock", "GECKO"),
        ("friday what did i tell you about the exam", "JOCASTA"),
        ("deploy my android app", "FORGE"),
        ("how are you", None),
        ("lol", None),
    ],
)
def test_a_request_reaches_whoever_holds_the_tools(
    text: str, expected: str | None
) -> None:
    """Asked to check his devices, FRIDAY made a joke about being a librarian.

    She has no security tools; EDITH does. A request nobody addressed should
    reach the operator whose patch it lands on rather than being deflected by
    the one who happens to be listening.
    """
    found = operators.for_domain(text)
    assert getattr(found, "name", None) == expected


def test_an_operator_must_say_what_is_missing_rather_than_joke() -> None:
    rule = operators.persona_rule(operators.addressed("edith, status?"))
    assert "no connected system for" in rule
    assert "name what it would need" in rule


def test_a_question_answered_by_arithmetic_is_not_a_search() -> None:
    """"When will it complete 1 light day" got "I couldn't find anything".

    True of the search and useless as an answer: the distance was on screen, the
    speed is known, and the rest is a division.
    """
    from friday.discord import tutor  # noqa: PLC0415

    tutor.remember("chan", "distance of voyager 1?", "about 22.9 billion km away")
    assert tutor.is_computation("chan", "When will it complete 1 light day") is True
    assert tutor.is_computation("chan", "in km") is True
    assert tutor.is_computation("chan", "how long until it gets there") is True

    # A number has to be in play, or it is just conversation.
    tutor._LAST.pop("empty", None)
    assert tutor.is_computation("empty", "how many people are coming") is False
    assert tutor.is_computation("chan", "lol that is wild") is False


def test_jarvis_is_a_real_roster_member_not_a_special_case() -> None:
    """He keeps the roster's invariants: own namespace, matching his name.

    "Full memory" is about what he can read, not where he writes. Pointing him
    at FRIDAY's namespace was the shortcut, and it broke the rule that
    namespaces are distinct — a rule worth more than the shortcut, since two
    operators writing into one corner is how memory gets confusing.
    """
    from friday.roster import ROSTER  # noqa: PLC0415

    jarvis = next(p for p in ROSTER.personas() if p.name == "JARVIS")
    assert jarvis.memory_namespace == "jarvis"
    edith = next(p for p in ROSTER.personas() if p.name == "EDITH")
    assert edith.memory_namespace == "edith"
    # Wide, but not the prime's set — nothing here locks anything down.
    assert len(jarvis.allowed_tools) > len(edith.allowed_tools)
    assert "security_lockdown" not in jarvis.allowed_tools


def test_jarvis_has_a_face_and_answers_to_his_name() -> None:
    from friday.discord import emblem  # noqa: PLC0415

    assert emblem.known("jarvis")
    assert (emblem.render("JARVIS") or b"").startswith(b"\x89PNG")
    assert getattr(operators.addressed("jarvis, status?"), "name", None) == "JARVIS"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("jarvis sitrep", True),
        ("catch me up", True),
        ("anything i should know", True),
        ("give me a rundown", True),
        ("jarvis how are you", False),          # ordinary conversation
        ("is my server secure", False),         # EDITH's, not his
    ],
)
def test_the_overview_is_his_and_specialities_are_not(
    text: str, expected: bool
) -> None:
    assert operators.wants_sitrep(text) is expected


def test_a_security_question_still_goes_to_edith_not_jarvis() -> None:
    """JARVIS is matched first, so he must not swallow the specialists."""
    assert getattr(
        operators.for_domain("friday check security of my devices"), "name", None
    ) == "EDITH"
    assert getattr(
        operators.for_domain("remind me at 6pm to call mum"), "name", None
    ) == "ORACLE"


@pytest.mark.anyio
async def test_a_sitrep_survives_every_source_being_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One dead token must not take the whole briefing with it."""

    async def _boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(operators, "machine_report", _boom)
    monkeypatch.setattr(operators, "github_report", _boom)
    monkeypatch.setattr(operators, "calendar_report", _boom)

    class App:
        state = type("S", (), {"settings": type("T", (), {
            "github_default_repo": ""})()})()

    from friday.discord import lookup  # noqa: PLC0415

    async def _quiet(question: str, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(lookup, "brief", _quiet)
    assert await operators.sitrep(App(), "sitrep") is None
