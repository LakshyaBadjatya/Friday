"""``/roster`` — the persona roster listing (Stage 2).

Exposes the FRIDAY persona roster — the prime (FRIDAY) plus the eight
least-privilege specialists (EDITH, ORACLE, GECKO, KAREN, VERONICA, JOCASTA,
VISION, FORGE) — as a read-only listing the dashboard / clients can render. Each
entry carries the persona's ``name`` (code-name), ``title`` (human role),
``scope`` (its least-privilege tool allow-list, sorted) and ``namespace`` (the
memory namespace it reads/writes under).

The roster adds no side-effecting surface, so it is **always available** (no
feature flag): it is a pure read over the
:class:`~friday.roster.RosterRegistry` wired on ``app.state`` at startup (with a
fall-back to the package-level canonical :data:`friday.roster.ROSTER` so a narrow
app build still answers). Nothing here reaches the network or an LLM SDK.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from friday.roster import ROSTER, Persona, RosterRegistry

router = APIRouter(prefix="/roster")


class PersonaView(BaseModel):
    """A serializable view of one roster persona for the ``/roster`` listing."""

    name: str
    title: str
    scope: list[str]
    namespace: str


class RosterResponse(BaseModel):
    """The ``GET /roster`` payload: every persona in declaration order.

    Declaration order puts the prime (FRIDAY) first, then the eight specialists,
    so the listing reads top-down from the broadest operator to the narrowest.
    """

    personas: list[PersonaView]
    count: int


def _roster(request: Request) -> RosterRegistry:
    """Return the shared :class:`RosterRegistry` from ``app.state``.

    Falls back to the package-level canonical :data:`friday.roster.ROSTER` when a
    caller built the app without wiring one (a narrow path), so the route always
    answers with the full roster.
    """
    roster = getattr(request.app.state, "roster", None)
    if isinstance(roster, RosterRegistry):
        return roster
    return ROSTER


def _view(persona: Persona) -> PersonaView:
    """Serialize one :class:`Persona` to its public listing view."""
    return PersonaView(
        name=persona.name,
        title=persona.title,
        scope=sorted(persona.allowed_tools),
        namespace=persona.memory_namespace,
    )


@router.get("", response_model=RosterResponse)
async def get_roster(request: Request) -> RosterResponse:
    """List FRIDAY plus the eight specialist personas (name/title/scope/namespace)."""
    personas = [_view(persona) for persona in _roster(request).personas()]
    return RosterResponse(personas=personas, count=len(personas))
