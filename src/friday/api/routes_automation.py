# © Lakshya Badjatya — Author
"""``/rules`` + ``/watchers`` — proactive evaluation services (Wave 4; default off).

Stateless surfaces over the pure proactive modules, each gated by its own flag
(every route 404s when off):

* ``POST /rules/evaluate {event, payload?, rules}`` (``FRIDAY_ENABLE_RULES``) —
  evaluate a set of IFTTT-style rules against one event and return the actions
  that fired. The caller supplies the rules + event (rules-as-a-service), so no
  event bus is assumed; *executing* a fired action stays a separate, broker-gated
  step.
* ``POST /watchers/conflicts {events}`` (``FRIDAY_ENABLE_WATCHERS``) — return the
  overlapping pairs in a set of calendar events.
* ``POST /watchers/price {price, above?, below?}`` (``FRIDAY_ENABLE_WATCHERS``) —
  whether a price breached a ceiling/floor.

Imports no LLM SDK — pure, deterministic evaluation over the injected request
data.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.proactive.rules import Rule, RulesEngine
from friday.proactive.watchers import TimeEvent, find_conflicts, price_breach

logger = get_logger("friday.api.routes_automation")

router = APIRouter()


def _enabled(request: Request, flag: str) -> bool:
    """Whether ``flag`` is set on the startup settings stashed on app state."""
    return bool(getattr(getattr(request.app.state, "settings", None), flag, False))


class RulesEvalRequest(BaseModel):
    """Body for ``POST /rules/evaluate``."""

    event: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)


class ConflictsRequest(BaseModel):
    """Body for ``POST /watchers/conflicts``."""

    events: list[TimeEvent] = Field(default_factory=list)


class PriceRequest(BaseModel):
    """Body for ``POST /watchers/price``."""

    price: float
    above: float | None = None
    below: float | None = None


@router.post("/rules/evaluate", response_model=None)
async def evaluate_rules(request: Request, body: RulesEvalRequest) -> JSONResponse:
    """Return the actions that fire for ``event``; 404 when the rules engine is off."""
    if not _enabled(request, "enable_rules"):
        return JSONResponse(status_code=404, content={"detail": "rules engine disabled"})
    fired = RulesEngine(body.rules).evaluate(body.event, body.payload)
    return JSONResponse(
        status_code=200, content={"fired": [f.model_dump() for f in fired]}
    )


@router.post("/watchers/conflicts", response_model=None)
async def watcher_conflicts(request: Request, body: ConflictsRequest) -> JSONResponse:
    """Return overlapping event pairs; 404 when watchers are off."""
    if not _enabled(request, "enable_watchers"):
        return JSONResponse(status_code=404, content={"detail": "watchers disabled"})
    conflicts = find_conflicts(body.events)
    return JSONResponse(
        status_code=200, content={"conflicts": [c.model_dump() for c in conflicts]}
    )


@router.post("/watchers/price", response_model=None)
async def watcher_price(request: Request, body: PriceRequest) -> JSONResponse:
    """Whether ``price`` breached a ceiling/floor; 404 when watchers are off."""
    if not _enabled(request, "enable_watchers"):
        return JSONResponse(status_code=404, content={"detail": "watchers disabled"})
    breach = price_breach(body.price, above=body.above, below=body.below)
    return JSONResponse(status_code=200, content={"breach": breach})
