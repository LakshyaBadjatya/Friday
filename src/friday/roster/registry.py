"""Lookup layer over the persona roster.

:class:`RosterRegistry` maps persona code-names to :class:`Persona` instances and
provides the access patterns the orchestrator/router need:

* :meth:`get` — case-insensitive lookup by name (raises ``KeyError`` if unknown);
* :meth:`by_intent` — coarse intent/domain keyword to the owning persona, with a
  fall-back to the prime for general/unknown intents;
* :meth:`for_tool` — every persona whose allow-list includes a given tool;
* :meth:`names` / :meth:`personas` — enumerate the roster.

The registry takes its personas as a constructor parameter (dependency
injection): it imports nothing from :mod:`friday.config` or :mod:`friday.app`.
The package-level :data:`friday.roster.ROSTER` is a pre-built instance over the
canonical :data:`~friday.roster.definitions.ROSTER_PERSONAS`.
"""

from __future__ import annotations

from collections.abc import Iterable

from friday.roster.definitions import INTENT_TO_PERSONA, Persona

# The persona returned when an intent is unknown / general. Kept here (not in the
# registry body) so it is a single named constant.
_PRIME_NAME = "FRIDAY"


class RosterRegistry:
    """An in-memory registry mapping persona code-names to :class:`Persona`.

    Args:
        personas: The personas to register, in order. Names are case-insensitive
            on lookup but stored canonically. Duplicate names (compared
            case-insensitively) raise :class:`ValueError`.
        intent_map: Optional override of the coarse intent/domain keyword to
            persona-name mapping used by :meth:`by_intent`. Defaults to the
            roster's canonical :data:`~friday.roster.definitions.INTENT_TO_PERSONA`.
        prime_name: The persona :meth:`by_intent` falls back to for unknown
            intents. Defaults to ``"FRIDAY"``.

    The registry preserves insertion order for :meth:`names` and
    :meth:`personas` so callers get a stable, declaration-ordered view.
    """

    def __init__(
        self,
        personas: Iterable[Persona],
        *,
        intent_map: dict[str, str] | None = None,
        prime_name: str = _PRIME_NAME,
    ) -> None:
        self._by_name: dict[str, Persona] = {}
        self._order: list[str] = []
        for persona in personas:
            key = persona.name.casefold()
            if key in self._by_name:
                raise ValueError(f"duplicate persona name: {persona.name!r}")
            self._by_name[key] = persona
            self._order.append(persona.name)
        self._intent_map: dict[str, str] = (
            INTENT_TO_PERSONA if intent_map is None else intent_map
        )
        self._prime_name = prime_name

    def get(self, name: str) -> Persona:
        """Return the persona named ``name`` (case-insensitive).

        Raises:
            KeyError: if no persona is registered under ``name``.
        """
        try:
            return self._by_name[name.casefold()]
        except KeyError:
            raise KeyError(f"unknown persona: {name!r}") from None

    def __contains__(self, name: object) -> bool:
        """Whether a persona is registered under ``name`` (case-insensitive)."""
        return isinstance(name, str) and name.casefold() in self._by_name

    def names(self) -> tuple[str, ...]:
        """The canonical persona names, in declaration order."""
        return tuple(self._order)

    def personas(self) -> tuple[Persona, ...]:
        """Every registered persona, in declaration order."""
        return tuple(self._by_name[n.casefold()] for n in self._order)

    def by_intent(self, intent: str) -> Persona:
        """Return the persona owning ``intent`` (case-insensitive).

        Unknown or general intents fall back to the prime persona. The empty
        string also falls back to the prime.
        """
        target = self._intent_map.get(intent.casefold(), self._prime_name)
        return self.get(target)

    def for_tool(self, tool_name: str) -> tuple[Persona, ...]:
        """Every persona whose ``allowed_tools`` includes ``tool_name``.

        Returned in declaration order. Empty if no persona declares the tool.
        """
        return tuple(
            self._by_name[n.casefold()]
            for n in self._order
            if tool_name in self._by_name[n.casefold()].allowed_tools
        )
