"""``POST /chat`` — the HTTP entrypoint into the core loop (Task 1.9).

Accepts ``{session_id, text}``, binds a per-request correlation id (so every log
line for the turn is traceable), drives the orchestrator, and returns
``{text, mode, route, audio}``. ``audio`` is always ``null`` this phase — voice
is a later flag.

Error mapping is honest and typed: a :class:`~friday.errors.FridayError` becomes
a clean JSON error body with a status that reflects the failure class —
:class:`~friday.errors.PermissionError` -> 403, :class:`~friday.errors.ProviderError`
-> 502, anything else in the family -> 400. The orchestrator already converts
most domain errors into in-character replies, so a raised ``FridayError`` here is
the exceptional path; we still map it cleanly rather than leaking a 500.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.errors import FridayError, PermissionError, ProviderError
from friday.logging import bind_correlation_id, get_logger
from friday.models.gateway import ModelGateway

logger = get_logger("friday.api.routes_chat")

router = APIRouter()


class ChatRequest(BaseModel):
    """Inbound chat turn.

    ``text`` and ``session_id`` are length-bounded so empty or pathologically
    large inputs are rejected by FastAPI with a 422 before any orchestration
    runs.
    """

    session_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8000)
    #: Optional per-turn model override (a ``provider:model`` catalog id). When
    #: set AND a :class:`~friday.models.gateway.ModelGateway` is the active LLM,
    #: this single turn is routed through the chosen model (the gateway's active
    #: model is restored afterwards). Omitted (``None``) keeps the default path
    #: exactly unchanged, so the single-provider/fake builds are untouched.
    model: str | None = Field(default=None, max_length=200)


class RouteView(BaseModel):
    """Serializable view of the router's decision for the response body."""

    mode: str
    agent: str | None
    rationale: str
    confidence: float


class ChatResponse(BaseModel):
    """Outbound chat turn."""

    text: str
    mode: str
    route: RouteView | None
    audio: None = None


class ErrorBody(BaseModel):
    """Typed error envelope for a mapped :class:`FridayError`."""

    error: str
    type: str


def _status_for(exc: FridayError) -> int:
    """Map a :class:`FridayError` subclass to an HTTP status code."""
    if isinstance(exc, PermissionError):
        return 403
    if isinstance(exc, ProviderError):
        return 502
    return 400


def _get_orchestrator(request: Request) -> Orchestrator:
    """Pull the process-wide orchestrator off app state (built at startup)."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not isinstance(orchestrator, Orchestrator):  # pragma: no cover - startup guard
        raise RuntimeError("orchestrator is not initialized on app.state")
    return orchestrator


def _get_gateway(request: Request) -> ModelGateway | None:
    """Return the process-wide :class:`ModelGateway`, or ``None`` when absent.

    A gateway is present only when ``app.py`` built one (OpenRouter/OpenCode keys
    or ``FRIDAY_LLM_PROVIDER == "gateway"``); on the single-provider/fake builds
    this is ``None`` and a ``model`` override is silently a no-op.
    """
    gateway = getattr(request.app.state, "gateway", None)
    return gateway if isinstance(gateway, ModelGateway) else None


async def _handle_with_model(
    request: Request,
    orchestrator: Orchestrator,
    state: GraphState,
    model: str | None,
) -> Any:
    """Drive the orchestrator, routing this one turn through ``model`` when set.

    When ``model`` is given AND a :class:`ModelGateway` backs the orchestrator's
    LLM, the gateway's active model is swapped to ``model`` for the duration of
    the turn and restored afterwards (try/finally), so the whole turn resolves to
    the chosen model without threading an override through the orchestrator. When
    ``model`` is ``None`` (the default) — or no gateway is wired — the orchestrator
    runs exactly as before, so the default path is unchanged.
    """
    gateway = _get_gateway(request)
    if model is None or gateway is None:
        return await orchestrator.handle(state)
    previous = gateway.active_model_id
    gateway.set_active(model)
    try:
        return await orchestrator.handle(state)
    finally:
        gateway.set_active(previous)


@router.post("/chat", response_model=None)
async def chat(request: Request, body: ChatRequest) -> JSONResponse:
    """Handle one chat turn end-to-end."""
    correlation_id = str(uuid.uuid4())
    bind_correlation_id(correlation_id)
    logger.info(
        "chat request received",
        extra={"session_id": body.session_id, "text_len": len(body.text)},
    )

    orchestrator = _get_orchestrator(request)
    state = GraphState(session_id=body.session_id, user_input=body.text)

    try:
        result = await _handle_with_model(request, orchestrator, state, body.model)
    except FridayError as exc:
        status = _status_for(exc)
        logger.warning(
            "chat turn raised FridayError",
            extra={"error_type": type(exc).__name__, "status": status},
        )
        return JSONResponse(
            status_code=status,
            content=ErrorBody(error=str(exc), type=type(exc).__name__).model_dump(),
        )

    route_view: RouteView | None = None
    if result.route is not None:
        route_view = RouteView(
            mode=result.route.mode.value,
            agent=result.route.agent,
            rationale=result.route.rationale,
            confidence=result.route.confidence,
        )

    payload: dict[str, Any] = ChatResponse(
        text=result.response or "",
        mode=result.mode.value,
        route=route_view,
    ).model_dump()
    return JSONResponse(status_code=200, content=payload)
