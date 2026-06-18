# © Lakshya Badjatya — Author
"""``/flows`` — the Flow Engine surface (Phase 1; default off).

Available ONLY when ``FRIDAY_ENABLE_FLOWS`` is set (``app.py`` then stashes a
:class:`~friday.flows.engine.FlowEngine` on ``app.state.flow_engine``). Off by
default every route is ``404``.

* ``POST /flows {goal}`` — decompose a goal into a flow and persist it (``planned``).
* ``POST /flows/{id}/run`` — drive the flow to a terminal state or a pause.
* ``GET  /flows`` — list flows (optionally ``?status=``).
* ``GET  /flows/{id}`` — one flow's current state + step traces.
* ``GET  /flows/{id}/events`` — the flow's audited transition events (from the
  hash-chained ledger, filtered to this flow).

Imports no LLM SDK — it drives the injected engine (whose planner degrades to a
single-step plan when the model is unavailable, so planning never hard-fails).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.flows.engine import FlowEngine
from friday.flows.models import FlowStatus
from friday.flows.templates import FlowTemplate
from friday.logging import get_logger

logger = get_logger("friday.api.routes_flows")

router = APIRouter()


class FlowRequest(BaseModel):
    """Inbound body for ``POST /flows`` — the goal to plan + run."""

    goal: str = Field(min_length=1, max_length=8000)


class RunRequest(BaseModel):
    """Inbound body for ``POST /flows/{id}/run`` — optional confirm flag.

    A side-effecting step the broker withholds pauses the flow; re-running with
    ``confirmed=true`` authorizes those irreversible actions to proceed.
    """

    confirmed: bool = False


def _get_engine(request: Request) -> FlowEngine | None:
    """The process-wide :class:`FlowEngine`, or ``None`` when the feature is off."""
    engine = getattr(request.app.state, "flow_engine", None)
    return engine if isinstance(engine, FlowEngine) else None


def _disabled() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "flows disabled"})


@router.post("/flows", response_model=None)
async def create_flow(request: Request, body: FlowRequest) -> JSONResponse:
    """Decompose ``goal`` into a persisted ``planned`` flow; 404 when disabled."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = await engine.plan(body.goal)
    logger.info("flow planned", extra={"flow_id": flow.id, "steps": len(flow.steps)})
    return JSONResponse(status_code=200, content=flow.model_dump(mode="json"))


@router.post("/flows/{flow_id}/run", response_model=None)
async def run_flow(
    request: Request, flow_id: str, body: RunRequest | None = None
) -> JSONResponse:
    """Drive the flow to a terminal state or a pause; 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    confirmed = body.confirmed if body is not None else False
    flow = await engine.run_by_id(flow_id, confirmed=confirmed)
    if flow is None:
        return JSONResponse(status_code=404, content={"detail": "flow not found"})
    logger.info("flow run", extra={"flow_id": flow.id, "status": flow.status.value})
    return JSONResponse(status_code=200, content=flow.model_dump(mode="json"))


@router.get("/flows", response_model=None)
async def list_flows(request: Request, status: str | None = None) -> JSONResponse:
    """List flows, optionally filtered by ``?status=``; 404 when disabled."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow_status = _parse_status(status)
    flows = engine.list_flows(flow_status)
    return JSONResponse(
        status_code=200,
        content={"flows": [f.model_dump(mode="json") for f in flows]},
    )


@router.get("/flows/{flow_id}", response_model=None)
async def get_flow(request: Request, flow_id: str) -> JSONResponse:
    """Return one flow's state + step traces; 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = engine.get(flow_id)
    if flow is None:
        return JSONResponse(status_code=404, content={"detail": "flow not found"})
    return JSONResponse(status_code=200, content=flow.model_dump(mode="json"))


@router.get("/flows/{flow_id}/events", response_model=None)
async def flow_events(request: Request, flow_id: str) -> JSONResponse:
    """Return the flow's audited transition events from the hash-chained ledger."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    ledger = getattr(request.app.state, "hash_audit", None)
    events: list[dict[str, Any]] = []
    if ledger is not None:
        for entry in ledger.entries():
            record = entry.record
            if record.get("flow_id") == flow_id:
                events.append(record)
    return JSONResponse(status_code=200, content={"events": events})


class ApproveRequest(BaseModel):
    """Inbound body for ``POST /flows/{id}/approve`` (both fields optional)."""

    step_id: str | None = None
    confirmed: bool = False


class SkipRequest(BaseModel):
    """Inbound body for ``POST /flows/{id}/skip`` — the step to skip."""

    step_id: str = Field(min_length=1)


class TemplateRequest(BaseModel):
    """Inbound body for ``POST /flow-templates`` — a reusable flow template."""

    name: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=8000)
    steps: list[dict[str, Any]] = Field(default_factory=list)


class TemplateRunRequest(BaseModel):
    """Inbound body for ``POST /flow-templates/{name}/run`` — substitution params."""

    params: dict[str, str] = Field(default_factory=dict)
    confirmed: bool = False


def _not_found() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "flow not found"})


def _ok(flow: Any) -> JSONResponse:
    return JSONResponse(status_code=200, content=flow.model_dump(mode="json"))


@router.post("/flows/{flow_id}/approve", response_model=None)
async def approve_flow(
    request: Request, flow_id: str, body: ApproveRequest | None = None
) -> JSONResponse:
    """Clear a step's approval gate and resume; 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    step_id = body.step_id if body is not None else None
    confirmed = body.confirmed if body is not None else False
    flow = await engine.approve(flow_id, step_id=step_id, confirmed=confirmed)
    return _not_found() if flow is None else _ok(flow)


@router.post("/flows/{flow_id}/cancel", response_model=None)
async def cancel_flow(request: Request, flow_id: str) -> JSONResponse:
    """Cancel a flow (terminal); 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = await engine.cancel(flow_id)
    return _not_found() if flow is None else _ok(flow)


@router.post("/flows/{flow_id}/pause", response_model=None)
async def pause_flow(request: Request, flow_id: str) -> JSONResponse:
    """Pause a flow (owner steering); 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = await engine.pause(flow_id)
    return _not_found() if flow is None else _ok(flow)


@router.post("/flows/{flow_id}/skip", response_model=None)
async def skip_step(
    request: Request, flow_id: str, body: SkipRequest
) -> JSONResponse:
    """Skip a still-pending step (owner steering); 404 when disabled/missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = await engine.skip(flow_id, body.step_id)
    return _not_found() if flow is None else _ok(flow)


@router.post("/flows/{flow_id}/simulate", response_model=None)
async def simulate_flow(request: Request, flow_id: str) -> JSONResponse:
    """Dry-run a flow (predict side effects, execute nothing); 404 when missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    flow = await engine.simulate(flow_id)
    return _not_found() if flow is None else _ok(flow)


@router.post("/flow-templates", response_model=None)
async def save_template(request: Request, body: TemplateRequest) -> JSONResponse:
    """Register a reusable flow template; 404 when flows are disabled."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    template = FlowTemplate(name=body.name, goal=body.goal, steps=body.steps)  # type: ignore[arg-type]
    saved = engine.save_template(template)
    if saved is None:
        return JSONResponse(status_code=404, content={"detail": "templates disabled"})
    return JSONResponse(status_code=200, content=saved.model_dump(mode="json"))


@router.get("/flow-templates", response_model=None)
async def list_templates(request: Request) -> JSONResponse:
    """List registered flow templates; 404 when flows are disabled."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    return JSONResponse(
        status_code=200,
        content={"templates": [t.model_dump(mode="json") for t in engine.list_templates()]},
    )


@router.post("/flow-templates/{name}/run", response_model=None)
async def run_template(
    request: Request, name: str, body: TemplateRunRequest | None = None
) -> JSONResponse:
    """Instantiate a template into a fresh flow and run it; 404 when missing."""
    engine = _get_engine(request)
    if engine is None:
        return _disabled()
    params = body.params if body is not None else {}
    confirmed = body.confirmed if body is not None else False
    flow = await engine.run_template(name, params, confirmed=confirmed)
    if flow is None:
        return JSONResponse(status_code=404, content={"detail": "template not found"})
    return _ok(flow)


def _parse_status(status: str | None) -> FlowStatus | None:
    """Coerce a ``?status=`` query value to a :class:`FlowStatus` (or ``None``)."""
    if not status:
        return None
    try:
        return FlowStatus(status)
    except ValueError:
        return None
