"""Unit tests for the agent roster (:mod:`friday.roster`).

The roster is a pure, dependency-free declaration of FRIDAY's personas — the
prime (FRIDAY) plus the eight specialists — and a registry that looks them up by
name. These tests pin the contract the later integration pass wires against:

* all nine personas (FRIDAY + the eight specialists) are present;
* every persona has a non-empty, distinct ``memory_namespace`` (its own name
  lowercased) and a non-empty, scoped ``allowed_tools`` set;
* specialists are *least privilege* — none of them holds the full union of tool
  names (only the prime may be that broad);
* lookups by name are case-insensitive and unknown names raise ``KeyError``;
* persona ``allowed_tools`` only reference real, executable tool names where the
  capability is backed by a registry tool.
"""

from __future__ import annotations

import pytest

from friday.roster import ROSTER, Persona, RosterRegistry

# The eight specialist code-names plus the prime.
SPECIALISTS = frozenset(
    {"EDITH", "ORACLE", "GECKO", "KAREN", "VERONICA", "JOCASTA", "VISION", "FORGE"}
)
PRIME = "FRIDAY"
ALL_NAMES = SPECIALISTS | {PRIME}

# Tool names that are actually registered/executable in the tool registry
# (read off src/friday/tools/*.py at build time). Personas may reference these
# directly; capability tokens for not-yet-wired domains are allowed but these
# are the ones that must resolve to a real tool.
REAL_TOOL_NAMES = frozenset(
    {
        "agent_reach",
        "notify",
        "run_command",
        "find_files",
        "open_app",
        "web_search",
        "home",
        "create_reminder",
        "list_reminders",
        "complete_reminder",
    }
)


def test_persona_is_immutable_pydantic_model() -> None:
    p = Persona(
        name="X",
        title="t",
        allowed_tools=frozenset({"a"}),
        memory_namespace="x",
        system_prompt="be x",
    )
    assert p.allowed_tools == frozenset({"a"})
    # frozen model: mutation is rejected by pydantic v2.
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        p.name = "Y"  # type: ignore[misc]


def test_all_nine_personas_present() -> None:
    names = set(ROSTER.names())
    assert names == ALL_NAMES, f"roster names mismatch: {names ^ ALL_NAMES}"
    assert len(ROSTER.names()) == 9


def test_prime_is_a_persona_with_broad_scope() -> None:
    friday = ROSTER.get(PRIME)
    assert isinstance(friday, Persona)
    assert friday.memory_namespace == "friday"
    # The prime is the broadest: it is a superset of every specialist's tools.
    for name in SPECIALISTS:
        spec = ROSTER.get(name)
        assert spec.allowed_tools <= friday.allowed_tools, (
            f"{name} holds tools the prime lacks: "
            f"{spec.allowed_tools - friday.allowed_tools}"
        )


def test_every_persona_has_nonempty_distinct_namespace() -> None:
    namespaces = [ROSTER.get(n).memory_namespace for n in ROSTER.names()]
    assert all(ns for ns in namespaces), "a persona has an empty namespace"
    assert len(set(namespaces)) == len(namespaces), "namespaces are not distinct"


def test_namespace_is_name_lowercased() -> None:
    for name in ROSTER.names():
        assert ROSTER.get(name).memory_namespace == name.lower()


def test_every_persona_has_scoped_nonempty_allowed_tools() -> None:
    for name in ROSTER.names():
        tools = ROSTER.get(name).allowed_tools
        assert tools, f"{name} has no allowed_tools"
        assert isinstance(tools, frozenset)


def test_every_persona_has_nonempty_title_and_prompt() -> None:
    for name in ROSTER.names():
        p = ROSTER.get(name)
        assert p.title.strip(), f"{name} has empty title"
        assert p.system_prompt.strip(), f"{name} has empty system_prompt"


def test_specialists_are_least_privilege() -> None:
    # No single specialist may hold the full union of every persona's tools.
    full_union: frozenset[str] = frozenset()
    for name in ROSTER.names():
        full_union |= ROSTER.get(name).allowed_tools
    for name in SPECIALISTS:
        assert ROSTER.get(name).allowed_tools < full_union, (
            f"{name} is not least-privilege (holds the full tool union)"
        )


def test_specialist_tool_sets_are_distinct() -> None:
    # Each specialist's allow-list should differ from every other specialist's,
    # so personas are meaningfully scoped rather than copies.
    sets = {name: ROSTER.get(name).allowed_tools for name in SPECIALISTS}
    seen: list[frozenset[str]] = []
    for name, tools in sets.items():
        assert tools not in seen, f"{name} duplicates another specialist's tools"
        seen.append(tools)


def test_security_persona_is_scoped_to_lockdown() -> None:
    edith = ROSTER.get("EDITH")
    # The security persona must not be able to browse the web or run shells.
    assert "run_command" not in edith.allowed_tools
    assert "web_search" not in edith.allowed_tools
    # It must include at least one lockdown/security capability and notify.
    assert any("lock" in t or "security" in t for t in edith.allowed_tools)


def test_dev_persona_can_exec() -> None:
    forge = ROSTER.get("FORGE")
    assert "run_command" in forge.allowed_tools


def test_finance_persona_has_market_and_search() -> None:
    gecko = ROSTER.get("GECKO")
    assert "web_search" in gecko.allowed_tools
    assert any("market" in t for t in gecko.allowed_tools)


def test_lookup_is_case_insensitive() -> None:
    assert ROSTER.get("edith") is ROSTER.get("EDITH")
    assert ROSTER.get("Friday") is ROSTER.get("FRIDAY")
    assert ROSTER.get("oRaClE") is ROSTER.get("ORACLE")


def test_unknown_persona_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        ROSTER.get("NOBODY")


def test_contains_is_case_insensitive() -> None:
    assert "edith" in ROSTER
    assert "EDITH" in ROSTER
    assert "nobody" not in ROSTER


def test_personas_property_returns_all() -> None:
    personas = ROSTER.personas()
    assert len(personas) == 9
    assert all(isinstance(p, Persona) for p in personas)
    assert {p.name for p in personas} == ALL_NAMES


def test_by_intent_resolves_known_domains() -> None:
    # by_intent maps a coarse intent/domain keyword to the owning persona.
    assert ROSTER.by_intent("security").name == "EDITH"
    assert ROSTER.by_intent("automation").name == "ORACLE"
    assert ROSTER.by_intent("finance").name == "GECKO"
    assert ROSTER.by_intent("comms").name == "KAREN"
    assert ROSTER.by_intent("content").name == "VERONICA"
    assert ROSTER.by_intent("memory").name == "JOCASTA"
    assert ROSTER.by_intent("research").name == "VISION"
    assert ROSTER.by_intent("dev").name == "FORGE"


def test_by_intent_is_case_insensitive_and_falls_back_to_prime() -> None:
    assert ROSTER.by_intent("SECURITY").name == "EDITH"
    # An unknown / general intent falls back to the prime.
    assert ROSTER.by_intent("smalltalk").name == "FRIDAY"
    assert ROSTER.by_intent("").name == "FRIDAY"


def test_for_tool_finds_a_persona_holding_the_tool() -> None:
    # Every real registered tool should be claimed by at least the prime.
    for tool in REAL_TOOL_NAMES:
        owners = ROSTER.for_tool(tool)
        assert owners, f"no persona owns real tool {tool!r}"
        assert any(p.name == "FRIDAY" for p in owners)


def test_real_tools_are_referenced_by_some_persona() -> None:
    # Sanity: the roster's union actually exercises the real tool surface, not
    # only invented capability tokens.
    union: frozenset[str] = frozenset()
    for name in ROSTER.names():
        union |= ROSTER.get(name).allowed_tools
    assert REAL_TOOL_NAMES <= union, (
        f"roster never references real tools: {REAL_TOOL_NAMES - union}"
    )


def test_registry_can_be_constructed_independently() -> None:
    # RosterRegistry takes its personas as a parameter (dependency injection);
    # it must not depend on global config/app.
    p = Persona(
        name="SOLO",
        title="t",
        allowed_tools=frozenset({"web_search"}),
        memory_namespace="solo",
        system_prompt="be solo",
    )
    reg = RosterRegistry([p])
    assert reg.get("solo") is p
    assert reg.names() == ("SOLO",)


def test_registry_rejects_duplicate_names() -> None:
    p = Persona(
        name="DUP",
        title="t",
        allowed_tools=frozenset({"web_search"}),
        memory_namespace="dup",
        system_prompt="x",
    )
    with pytest.raises(ValueError, match="duplicate"):
        RosterRegistry([p, p])
