"""``/admin`` — the observability + control-plane API (Phase 5, Stage 2A).

This router exposes the Stage-1 observability stores (held on ``app.state`` by the
lifespan in :mod:`friday.app`) and a small runtime control surface that the
Streamlit dashboard (Stage 2B) consumes. The gate verifies *this* API — the JSON
the dashboard reads — not the UI itself.

Endpoints (all under the ``/admin`` prefix):

* ``GET  /admin/flags``   — the effective feature flags: the settings defaults
  with any in-memory runtime overrides applied on top.
* ``POST /admin/flags``   — set one runtime flag override (``{name, value}``) on a
  mutable holder on ``app.state`` and return the new effective flags. Unknown
  flag names are rejected ``400`` (a typo can't silently create a phantom flag).
* ``GET  /admin/state``   — live state: the active sessions (id + current mode +
  short-term buffer size) plus durable-memory stats.
* ``GET  /admin/audit``   — recent tool-call audit rows (sensitive args redacted)
  plus the recent durable security/audit rows when a long-term store is wired.
* ``GET  /admin/traces``  — recent per-request traces (each correlation id with
  its span names + timings) — the evidence that every turn is traced.
* ``GET  /admin/metrics`` — the :meth:`~friday.observability.metrics.Metrics.snapshot`.

Everything is read off ``app.state``; the router constructs nothing heavy and
never reaches the network or an LLM SDK. The holders are populated at startup, so
a fresh app (even before any ``/chat``) answers these with empty-but-valid views.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.config import Settings, get_settings
from friday.observability.audit import AuditLog, ToolCallAudit
from friday.observability.metrics import Metrics
from friday.observability.tracing import Tracer

router = APIRouter(prefix="/admin")

# The settings fields treated as runtime-toggleable feature flags. Only these
# names are accepted by ``POST /admin/flags`` (a closed set, so a typo can't
# mint a phantom flag), and they are the keys ``GET /admin/flags`` reports.
_FEATURE_FLAGS: tuple[str, ...] = (
    "enable_voice",
    "enable_home",
    "memory_autowrite",
    "alert_dedupe",
    "log_json",
)


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class FlagsResponse(BaseModel):
    """The effective feature flags (settings defaults + runtime overrides)."""

    flags: dict[str, bool]


class FlagUpdate(BaseModel):
    """A single runtime flag override request body."""

    name: str = Field(min_length=1, max_length=100)
    value: bool


class SessionState(BaseModel):
    """Live state for one active session."""

    session_id: str
    mode: str | None
    short_term_size: int


class StateResponse(BaseModel):
    """The ``GET /admin/state`` payload: active sessions + memory stats."""

    sessions: list[SessionState]
    memory: dict[str, Any]


class AuditResponse(BaseModel):
    """Recent tool-call audit rows + recent durable security/audit rows."""

    tool_calls: list[ToolCallAudit]
    security: list[dict[str, Any]]


class TraceView(BaseModel):
    """A serializable view of one recent trace for the traces endpoint."""

    correlation_id: str
    mode: str | None
    started: float
    spans: list[dict[str, Any]]


class TracesResponse(BaseModel):
    """Recent traces, oldest-first (newest last)."""

    traces: list[TraceView]


# --------------------------------------------------------------------------- #
# app.state accessors (read-only, with safe empty fallbacks)
# --------------------------------------------------------------------------- #
def _flag_overrides(request: Request) -> dict[str, bool]:
    """Return the mutable runtime flag-override holder from ``app.state``.

    The holder is created in the lifespan; if a caller built the app without it
    (a narrow path), an empty dict is created on the fly so the route still works.
    """
    overrides = getattr(request.app.state, "flag_overrides", None)
    if not isinstance(overrides, dict):
        overrides = {}
        request.app.state.flag_overrides = overrides
    return overrides


def _settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def _tracer(request: Request) -> Tracer | None:
    tracer = getattr(request.app.state, "tracer", None)
    return tracer if isinstance(tracer, Tracer) else None


def _audit(request: Request) -> AuditLog | None:
    audit = getattr(request.app.state, "audit", None)
    return audit if isinstance(audit, AuditLog) else None


def _metrics(request: Request) -> Metrics | None:
    metrics = getattr(request.app.state, "metrics", None)
    return metrics if isinstance(metrics, Metrics) else None


def _effective_flags(settings: Settings, overrides: dict[str, bool]) -> dict[str, bool]:
    """Compute the effective flags: settings defaults with overrides applied."""
    flags: dict[str, bool] = {}
    for name in _FEATURE_FLAGS:
        if name in overrides:
            flags[name] = overrides[name]
        else:
            flags[name] = bool(getattr(settings, name))
    return flags


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/flags", response_model=FlagsResponse)
async def get_flags(request: Request) -> FlagsResponse:
    """Return the effective feature flags (defaults + runtime overrides)."""
    return FlagsResponse(
        flags=_effective_flags(_settings(request), _flag_overrides(request))
    )


@router.post("/flags", response_model=None)
async def set_flag(request: Request, body: FlagUpdate) -> JSONResponse:
    """Set one runtime flag override and return the new effective flags.

    Only the closed :data:`_FEATURE_FLAGS` set is accepted; an unknown name is a
    ``400`` so a typo cannot silently create a phantom flag the rest of the system
    never reads.
    """
    if body.name not in _FEATURE_FLAGS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"unknown feature flag {body.name!r}",
                "type": "UnknownFlag",
            },
        )
    overrides = _flag_overrides(request)
    overrides[body.name] = body.value
    payload = FlagsResponse(
        flags=_effective_flags(_settings(request), overrides)
    ).model_dump()
    return JSONResponse(status_code=200, content=payload)


@router.get("/state", response_model=StateResponse)
async def get_state(request: Request) -> StateResponse:
    """Return live per-session state and durable-memory stats.

    Active sessions and their current mode are derived from the most-recent
    traces (each trace's ``correlation_id`` is the session id, stamped with the
    turn's mode); the short-term buffer size is read from the shared
    :class:`~friday.memory.short_term.ShortTermMemory`. Memory stats come from the
    long-term store when one is wired.
    """
    sessions: list[SessionState] = []
    seen: dict[str, str | None] = {}
    tracer = _tracer(request)
    if tracer is not None:
        # Walk newest-first so the latest mode per session wins.
        for trace in reversed(tracer.recent(256)):
            if trace.correlation_id not in seen:
                seen[trace.correlation_id] = trace.mode

    short_term = getattr(request.app.state, "short_term", None)
    for session_id, mode in seen.items():
        size = 0
        if short_term is not None:
            size = len(short_term.history(session_id))
        sessions.append(
            SessionState(session_id=session_id, mode=mode, short_term_size=size)
        )

    return StateResponse(sessions=sessions, memory=_memory_stats(request))


@router.get("/audit", response_model=AuditResponse)
async def get_audit(request: Request) -> AuditResponse:
    """Return recent tool-call audit rows + recent durable security/audit rows."""
    audit = _audit(request)
    tool_calls = audit.recent(100) if audit is not None else []
    return AuditResponse(tool_calls=tool_calls, security=_security_audit(request))


@router.get("/traces", response_model=TracesResponse)
async def get_traces(request: Request) -> TracesResponse:
    """Return recent traces, each with its correlation id, mode, and spans."""
    tracer = _tracer(request)
    views: list[TraceView] = []
    if tracer is not None:
        for trace in tracer.recent(100):
            views.append(
                TraceView(
                    correlation_id=trace.correlation_id,
                    mode=trace.mode,
                    started=trace.started,
                    spans=[span.model_dump() for span in trace.spans],
                )
            )
    return TracesResponse(traces=views)


@router.get("/metrics", response_model=None)
async def get_metrics(request: Request) -> JSONResponse:
    """Return the :class:`~friday.observability.metrics.Metrics` snapshot."""
    metrics = _metrics(request)
    snap = (
        metrics.snapshot()
        if metrics is not None
        else {"requests": 0, "tool_calls": 0, "errors": 0, "by_mode": {}}
    )
    return JSONResponse(status_code=200, content=snap)


# --------------------------------------------------------------------------- #
# Durable-store views (graceful when a store is not wired)
# --------------------------------------------------------------------------- #
def _memory_stats(request: Request) -> dict[str, Any]:
    """Return coarse durable-memory counts; empties when no store is wired."""
    long_term = getattr(request.app.state, "long_term", None)
    if long_term is None:
        return {"facts": 0, "tasks": 0, "audit": 0}
    return {
        "facts": len(long_term.query_facts("", limit=10_000)),
        "tasks": len(long_term.task_history(limit=10_000)),
        "audit": len(long_term.audit_history(limit=10_000)),
    }


def _security_audit(request: Request) -> list[dict[str, Any]]:
    """Return recent durable security/audit rows when a long-term store is wired."""
    long_term = getattr(request.app.state, "long_term", None)
    if long_term is None:
        return []
    return [row.model_dump() for row in long_term.audit_history(limit=100)]
