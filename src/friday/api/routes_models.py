# © Lakshya Badjatya — Author
"""``/models`` — the multi-model gateway control surface.

Available ONLY when a :class:`~friday.models.gateway.ModelGateway` was built at
startup (``app.py`` wires it onto ``app.state.gateway`` when OpenRouter/OpenCode
keys are present or ``FRIDAY_LLM_PROVIDER == "gateway"``). With no gateway —
e.g. the offline ``fake`` build — every route here is ``404`` so the feature
simply does not exist for callers (mirroring the flagged Tier-1/2 routes).

Three surfaces, all read off the shared gateway + catalog:

* ``GET /models`` -> ``{"active": <id>, "models": [ModelInfo...]}`` — the active
  model id plus the catalogued models the build can actually serve.
* ``POST /models/active {model_id}`` -> ``{"active": <id>}`` — switch the active
  model; an unknown id is ``404`` (it is not in the catalog) so the caller learns
  the switch did not take.
* ``POST /models/compare {prompt, models?, judge?}`` -> ``{"results": [...],
  "best": <id|null>}`` — fan a single user prompt out across several models
  (defaulting to ``settings.compare_model_ids``), optionally asking a judge model
  to name the best answer. Compare never raises per-model (each failure is
  captured in its :class:`~friday.models.gateway.CompareResult`).

This module imports no LLM SDK: it depends only on the gateway/catalog, so the
``openai`` import stays confined to :mod:`friday.providers.llm`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.models.catalog import ModelCatalog
from friday.models.gateway import ModelGateway
from friday.providers.llm import Message

logger = get_logger("friday.api.routes_models")

router = APIRouter()


class SetActiveRequest(BaseModel):
    """Inbound body for ``POST /models/active`` — the id to make active."""

    model_id: str = Field(min_length=1, max_length=200)


class CompareRequest(BaseModel):
    """Inbound body for ``POST /models/compare``.

    ``models`` defaults to the configured ``compare_model_ids`` when omitted (or
    empty), so a bare ``{"prompt": ...}`` fans out to the build's default compare
    set. ``judge`` opts into the non-fatal LLM-judge pass that names the best
    answer.
    """

    prompt: str = Field(min_length=1, max_length=8000)
    models: list[str] = Field(default_factory=list)
    judge: bool = False


def _get_gateway(request: Request) -> ModelGateway | None:
    """Return the process-wide :class:`ModelGateway`, or ``None`` when absent.

    ``None`` means no gateway was built at startup (the fake/single-provider
    build), in which case every route here answers ``404``.
    """
    gateway = getattr(request.app.state, "gateway", None)
    return gateway if isinstance(gateway, ModelGateway) else None


def _get_catalog(request: Request) -> ModelCatalog | None:
    """Return the process-wide :class:`ModelCatalog`, or ``None`` when absent."""
    catalog = getattr(request.app.state, "model_catalog", None)
    return catalog if isinstance(catalog, ModelCatalog) else None


def _disabled() -> JSONResponse:
    """The canonical ``model gateway disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "model gateway disabled"})


def _compare_model_ids(request: Request) -> list[str]:
    """The configured default compare set, read off the startup settings."""
    settings = getattr(request.app.state, "settings", None)
    ids = getattr(settings, "compare_model_ids", None)
    return list(ids) if ids else []


@router.get("/models", response_model=None)
async def list_models(request: Request) -> JSONResponse:
    """List the available catalogued models + the active id; 404 with no gateway."""
    gateway = _get_gateway(request)
    catalog = _get_catalog(request)
    if gateway is None or catalog is None:
        return _disabled()
    models = [info.model_dump() for info in catalog.list_models()]
    return JSONResponse(
        status_code=200,
        content={"active": gateway.active_model_id, "models": models},
    )


@router.post("/models/active", response_model=None)
async def set_active(request: Request, body: SetActiveRequest) -> JSONResponse:
    """Switch the active model; 404 with no gateway, 404 for an unknown id."""
    gateway = _get_gateway(request)
    catalog = _get_catalog(request)
    if gateway is None or catalog is None:
        return _disabled()
    if catalog.get(body.model_id) is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"unknown model id: {body.model_id}"},
        )
    gateway.set_active(body.model_id)
    logger.info("active model switched", extra={"model_id": body.model_id})
    return JSONResponse(status_code=200, content={"active": body.model_id})


@router.post("/models/compare", response_model=None)
async def compare_models(request: Request, body: CompareRequest) -> JSONResponse:
    """Fan a prompt out across models, optionally judging; 404 with no gateway."""
    gateway = _get_gateway(request)
    if gateway is None:
        return _disabled()
    model_ids = body.models or _compare_model_ids(request)
    messages = [Message(role="user", content=body.prompt)]
    results = await gateway.compare(messages, model_ids)
    best: str | None = None
    if body.judge:
        best = await gateway.judge(
            body.prompt, results, judge_model_id=gateway.active_model_id
        )
    payload: dict[str, Any] = {
        "results": [r.model_dump() for r in results],
        "best": best,
    }
    return JSONResponse(status_code=200, content=payload)
