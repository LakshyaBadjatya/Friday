# © Lakshya Badjatya — Author
"""``/ensemble`` — the multi-operator debate surface (Wave 1; default off).

Available ONLY when ``FRIDAY_ENABLE_ENSEMBLE`` is set (``app.py`` then stashes an
:class:`~friday.core.ensemble.Ensemble` on ``app.state.ensemble``). Off by default
every route here is ``404`` so the feature simply does not exist for callers
(mirroring the other flagged Tier-1/2 surfaces).

* ``POST /ensemble/debate {question, operators?}`` -> the per-operator drafts plus
  the fused synthesis. ``operators`` is an optional list of roster code-names; when
  omitted a default panel (VISION / GECKO / JOCASTA) debates. Unknown names are
  dropped; if none remain the request is ``400``.

Imports no LLM SDK — it drives the injected :class:`Ensemble` (which depends only
on the ``LLMProvider`` contract).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.core.ensemble import Ensemble
from friday.logging import get_logger
from friday.roster.definitions import ROSTER_PERSONAS

logger = get_logger("friday.api.routes_ensemble")

router = APIRouter()

#: The default debate panel when the caller names no operators.
_DEFAULT_PANEL: tuple[str, ...] = ("VISION", "GECKO", "JOCASTA")


class DebateRequest(BaseModel):
    """Inbound body for ``POST /ensemble/debate``."""

    question: str = Field(min_length=1, max_length=8000)
    operators: list[str] = Field(default_factory=list)


def _persona_prompts() -> dict[str, str]:
    """Map each roster code-name (upper-cased) to its system prompt."""
    return {p.name.upper(): p.system_prompt for p in ROSTER_PERSONAS}


def _get_ensemble(request: Request) -> Ensemble | None:
    """The process-wide :class:`Ensemble`, or ``None`` when the feature is off."""
    ensemble = getattr(request.app.state, "ensemble", None)
    return ensemble if isinstance(ensemble, Ensemble) else None


@router.post("/ensemble/debate", response_model=None)
async def debate(request: Request, body: DebateRequest) -> JSONResponse:
    """Run one operator debate; 404 when disabled, 400 when no operator is known."""
    ensemble = _get_ensemble(request)
    if ensemble is None:
        return JSONResponse(status_code=404, content={"detail": "ensemble disabled"})
    prompts = _persona_prompts()
    names = [n.upper() for n in body.operators] or list(_DEFAULT_PANEL)
    operators = [(name, prompts[name]) for name in names if name in prompts]
    if not operators:
        return JSONResponse(
            status_code=400, content={"detail": "no known operators named"}
        )
    result = await ensemble.debate(body.question, operators)
    logger.info(
        "ensemble debate", extra={"operators": len(operators)}
    )
    return JSONResponse(status_code=200, content=result.model_dump())
