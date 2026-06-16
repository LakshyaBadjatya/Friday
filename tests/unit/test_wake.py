# © Lakshya Badjatya — Author
"""Unit tests for wake/summon command parsing + per-operator voice styles."""

from __future__ import annotations

from friday.voice.wake import OPERATOR_VOICES, parse_wake_command, voice_for

_OPERATORS = ["FRIDAY", "EDITH", "ORACLE", "GECKO", "VISION"]


def test_wake_phrasings() -> None:
    for phrase in ("Hey Friday", "hey friday", "hi Friday", "ok friday, you there?"):
        cmd = parse_wake_command(phrase, _OPERATORS)
        assert cmd is not None and cmd.kind == "wake" and cmd.operator is None


def test_summon_resolves_operator() -> None:
    for verb in ("summon", "spawn", "call", "get", "wake"):
        cmd = parse_wake_command(f"Friday {verb} Vision", _OPERATORS)
        assert cmd is not None and cmd.kind == "summon" and cmd.operator == "VISION"


def test_summon_is_case_insensitive() -> None:
    cmd = parse_wake_command("friday summon gecko", _OPERATORS)
    assert cmd is not None and cmd.operator == "GECKO"


def test_summon_unknown_operator_is_not_a_command() -> None:
    assert parse_wake_command("Friday summon Batman", _OPERATORS) is None


def test_non_commands_return_none() -> None:
    assert parse_wake_command("", _OPERATORS) is None
    assert parse_wake_command("what's the weather", _OPERATORS) is None
    assert parse_wake_command("tell me about fridays", _OPERATORS) is None


def test_summon_beats_wake_when_both_present() -> None:
    cmd = parse_wake_command("hey friday, summon edith", _OPERATORS)
    assert cmd is not None and cmd.kind == "summon" and cmd.operator == "EDITH"


def test_voice_styles_are_distinct_per_operator() -> None:
    # Every roster operator has a voice, and they are not all identical.
    assert len(OPERATOR_VOICES) == 9
    signatures = {(v.pitch, v.rate, v.hint) for v in OPERATOR_VOICES.values()}
    assert len(signatures) >= 8  # near-unique timbres
    assert voice_for("VISION").hint == "male"
    assert voice_for("nobody").pitch == 1.0  # neutral default for unknown
