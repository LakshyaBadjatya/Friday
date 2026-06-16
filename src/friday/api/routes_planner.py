# © Lakshya Badjatya — Author
"""``/planner`` — the task-decomposition surface (Wave 1; default off).

Available ONLY when ``FRIDAY_ENABLE_PLANNER`` is set (``app.py`` then stashes a
:class:`~friday.core.planner.Planner` on ``app.state.planner``). Off by default
the route is ``404``.

* ``POST /planner/plan {goal}`` -> ``{"plan": {...}, "rendered": "..."}`` — decompose
  a goal into a DAG of steps and return both the structured plan and a
  human-readable rendering for confirmation. This endpoint only *plans*; executing
  the steps (through the broker, confirm-gated) is a separate, deliberate action.

Imports no LLM SDK — it drives the injected :class:`Planner` (which degrades to a
single-step plan if the model is unavailable, so the route never fails on that).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.core.planner import Planner
from friday.logging import get_logger

logger = get_logger("friday.api.routes_planner")

router = APIRouter()


class PlanRequest(BaseModel):
    """Inbound body for ``POST /planner/plan`` — the goal to decompose."""

    goal: str = Field(min_length=1, max_length=8000)


def _get_planner(request: Request) -> Planner | None:
    """The process-wide :class:`Planner`, or ``None`` when the feature is off."""
    planner = getattr(request.app.state, "planner", None)
    return planner if isinstance(planner, Planner) else None


@router.post("/planner/plan", response_model=None)
async def make_plan(request: Request, body: PlanRequest) -> JSONResponse:
    """Decompose ``goal`` into a plan; 404 when disabled."""
    planner = _get_planner(request)
    if planner is None:
        return JSONResponse(status_code=404, content={"detail": "planner disabled"})
    plan = await planner.decompose(body.goal)
    logger.info("planner decompose", extra={"steps": len(plan.steps)})
    return JSONResponse(
        status_code=200,
        content={"plan": plan.model_dump(), "rendered": plan.render()},
    )
