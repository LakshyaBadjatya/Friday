"""FRIDAY's persona roster: the prime plus eight least-privilege specialists.

Public surface (what the integration pass wires):

* :class:`Persona` — the frozen, dependency-free persona value object.
* :class:`RosterRegistry` — case-insensitive lookup over a set of personas.
* :data:`ROSTER` — a pre-built :class:`RosterRegistry` over the canonical
  roster (FRIDAY + EDITH, ORACLE, GECKO, KAREN, VERONICA, JOCASTA, VISION,
  FORGE).

This package imports nothing from :mod:`friday.config` or :mod:`friday.app`;
personas are pure data and the registry takes its personas as a parameter, so
the orchestrator/router can inject or override the roster at integration time.
"""

from __future__ import annotations

from friday.roster.definitions import ROSTER_PERSONAS, Persona
from friday.roster.registry import RosterRegistry

# The canonical, pre-built roster registry over all nine personas.
ROSTER: RosterRegistry = RosterRegistry(ROSTER_PERSONAS)

__all__ = ["ROSTER", "Persona", "RosterRegistry"]
