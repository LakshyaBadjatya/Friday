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
    """A serializable view of one roster persona for the ``/roster`` listing.

    ``model`` / ``model_label`` carry the persona's assigned free model (a
    ``provider:model`` catalog id and its display label) when one is wired; both
    are ``None`` on builds without a multi-model gateway, so a client can render
    "EDITH · GPT-OSS 20B" when present and just the operator otherwise.
    """

    name: str
    title: str
    scope: list[str]
    namespace: str
    model: str | None = None
    model_label: str | None = None


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


def _persona_models(request: Request) -> dict[str, str]:
    """The persona -> model-id assignment from ``app.state`` (empty when unwired)."""
    mapping = getattr(request.app.state, "persona_models", None)
    return mapping if isinstance(mapping, dict) else {}


def _model_label(request: Request, model_id: str | None) -> str | None:
    """Resolve a catalog id to its display label via ``app.state.model_catalog``."""
    if model_id is None:
        return None
    catalog = getattr(request.app.state, "model_catalog", None)
    if catalog is None:
        return None
    info = catalog.get(model_id)
    return info.label if info is not None else None


def _view(
    persona: Persona, model_id: str | None, model_label: str | None
) -> PersonaView:
    """Serialize one :class:`Persona` (plus its assigned model) to its listing view."""
    return PersonaView(
        name=persona.name,
        title=persona.title,
        scope=sorted(persona.allowed_tools),
        namespace=persona.memory_namespace,
        model=model_id,
        model_label=model_label,
    )


@router.get("", response_model=RosterResponse)
async def get_roster(request: Request) -> RosterResponse:
    """List FRIDAY plus the eight specialists (name/title/scope/namespace/model)."""
    models = _persona_models(request)
    personas: list[PersonaView] = []
    for persona in _roster(request).personas():
        model_id = models.get(persona.name)
        personas.append(_view(persona, model_id, _model_label(request, model_id)))
    return RosterResponse(personas=personas, count=len(personas))
