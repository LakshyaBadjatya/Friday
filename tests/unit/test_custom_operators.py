# © Lakshya Badjatya — Author
"""Unit tests for user-declared custom operators (:mod:`friday.roster.custom`).

A custom operator is the SAME frozen :class:`Persona` value object the built-ins
use, declared by the owner in a compact pipe-delimited mini-format::

    NAME|Title|tool1,tool2|namespace|system prompt text

These tests pin the contract the integration pass wires against:

* a well-formed entry parses into one :class:`Persona`, trimming whitespace and
  comma-splitting the tool allow-list;
* each malformed shape (too few fields, empty name, empty tools) raises a clear
  :class:`ValueError`, and the existing :class:`Persona` validators enforce the
  rest (empty title / prompt);
* the namespace defaults to ``name.lower()`` when the fourth field is empty;
* blank entries are skipped;
* :func:`merge_personas` lets the built-ins win a case-insensitive name
  collision (the colliding custom is dropped) and appends the rest in order, so a
  clean custom operator is resolvable via a :class:`RosterRegistry`.
"""

from __future__ import annotations

import pytest

from friday.roster import RosterRegistry
from friday.roster.custom import merge_personas, parse_custom_operators
from friday.roster.definitions import ROSTER_PERSONAS, Persona


def test_parses_a_valid_entry() -> None:
    [persona] = parse_custom_operators(
        ["SCOUT|Field Recon|web_search, notify|recon|You are SCOUT, the scout."]
    )
    assert isinstance(persona, Persona)
    assert persona.name == "SCOUT"
    assert persona.title == "Field Recon"
    # The tool field is comma-split and whitespace-trimmed into the allow-list.
    assert persona.allowed_tools == frozenset({"web_search", "notify"})
    assert persona.memory_namespace == "recon"
    assert persona.system_prompt == "You are SCOUT, the scout."


def test_too_few_fields_raises() -> None:
    with pytest.raises(ValueError, match="pipe-delimited fields"):
        parse_custom_operators(["SCOUT|Field Recon|web_search|recon"])


def test_too_many_fields_raises() -> None:
    with pytest.raises(ValueError, match="pipe-delimited fields"):
        parse_custom_operators(
            ["SCOUT|Field Recon|web_search|recon|prompt|extra"]
        )


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="empty name"):
        parse_custom_operators(["|Field Recon|web_search|recon|be scout"])


def test_empty_tool_list_raises() -> None:
    with pytest.raises(ValueError, match="empty tool allow-list"):
        parse_custom_operators(["SCOUT|Field Recon| , |recon|be scout"])


def test_empty_title_raises_via_persona_validator() -> None:
    # The Persona model (not this parser) enforces a non-empty title.
    with pytest.raises(ValueError):
        parse_custom_operators(["SCOUT||web_search|recon|be scout"])


def test_empty_prompt_raises_via_persona_validator() -> None:
    # The Persona model enforces a non-empty system prompt.
    with pytest.raises(ValueError):
        parse_custom_operators(["SCOUT|Field Recon|web_search|recon|"])


def test_namespace_defaults_to_name_lowercased() -> None:
    [persona] = parse_custom_operators(
        ["Scout|Field Recon|web_search||be scout"]
    )
    assert persona.memory_namespace == "scout"


def test_blank_entries_are_skipped() -> None:
    personas = parse_custom_operators(
        ["", "   ", "SCOUT|Field Recon|web_search|recon|be scout"]
    )
    assert [p.name for p in personas] == ["SCOUT"]


def test_collision_with_builtin_is_dropped() -> None:
    # A custom "EDITH" (any case) collides with the built-in and is dropped;
    # the built-in EDITH wins.
    [custom_edith] = parse_custom_operators(
        ["edith|Impostor|web_search|edith|not the real edith"]
    )
    merged = merge_personas(ROSTER_PERSONAS, [custom_edith])
    assert merged == ROSTER_PERSONAS
    # The surviving EDITH is the built-in (lockdown-scoped), not the impostor.
    by_name = {p.name: p for p in merged}
    assert "web_search" not in by_name["EDITH"].allowed_tools


def test_clean_custom_is_appended_and_resolvable() -> None:
    [scout] = parse_custom_operators(
        ["SCOUT|Field Recon|web_search,notify|recon|be scout"]
    )
    merged = merge_personas(ROSTER_PERSONAS, [scout])
    # Appended after every built-in, in order.
    assert len(merged) == len(ROSTER_PERSONAS) + 1
    assert merged[: len(ROSTER_PERSONAS)] == ROSTER_PERSONAS
    assert merged[-1] is scout
    # And resolvable (case-insensitively) via a registry built over the merge.
    registry = RosterRegistry(merged)
    assert registry.get("scout") is scout
    assert "SCOUT" in registry


def test_multiple_customs_preserve_order() -> None:
    customs = parse_custom_operators(
        [
            "ALPHA|First|web_search|alpha|be alpha",
            "BETA|Second|notify|beta|be beta",
        ]
    )
    merged = merge_personas(ROSTER_PERSONAS, customs)
    assert [p.name for p in merged[len(ROSTER_PERSONAS) :]] == ["ALPHA", "BETA"]
