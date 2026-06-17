# © Lakshya Badjatya — Author
"""``/security`` — secret-rotation reminders + audit anchoring (Wave 2; default off).

Each surface is gated by its own flag (404 when off):

* ``POST /security/rotation {secrets, max_age_seconds}`` (``FRIDAY_ENABLE_SECRET_ROTATION``)
  — report which of the posted secrets (name + last-rotated timestamp) are overdue.
* ``POST /security/anchor`` (``FRIDAY_ENABLE_AUDIT_ANCHOR``) — pin the audit ledger's
  current head hash out-of-band by appending a timestamped anchor to
  ``audit_anchor_path``; later, the anchored hash being absent from the live chain
  is evidence the ledger was rewritten. Reads the shared hash-chained ledger off
  ``app.state``.

Imports no LLM SDK — pure policy/IO over the injected request data and ledger.
"""

from __future__ import annotations

import time
from pathlib import Path

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.broker import HashChainedAudit
from friday.broker.audit import GENESIS_HASH
from friday.logging import get_logger
from friday.security.anchor import make_anchor
from friday.security.rotation import RotationPolicy, SecretAge

logger = get_logger("friday.api.routes_security")

router = APIRouter()


def _enabled(request: Request, flag: str) -> bool:
    """Whether ``flag`` is set on the startup settings stashed on app state."""
    return bool(getattr(getattr(request.app.state, "settings", None), flag, False))


class RotationRequest(BaseModel):
    """Body for ``POST /security/rotation``."""

    secrets: list[SecretAge] = Field(default_factory=list)
    max_age_seconds: float = Field(gt=0)


@router.post("/security/rotation", response_model=None)
async def rotation_due(request: Request) -> JSONResponse:
    """Return the names of the posted secrets overdue for rotation; 404 when off.

    The body is parsed *after* the feature-flag check (not via a bound parameter)
    so a disabled feature returns 404 even for a malformed body — a bound param
    would let FastAPI 422 first, leaking that the route exists.
    """
    if not _enabled(request, "enable_secret_rotation"):
        return JSONResponse(
            status_code=404, content={"detail": "secret rotation disabled"}
        )
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "invalid body"})
    try:
        body = RotationRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    policy = RotationPolicy(body.max_age_seconds)
    due = policy.due(body.secrets, now=time.time())
    return JSONResponse(status_code=200, content={"due": due})


@router.post("/security/anchor", response_model=None)
async def anchor_ledger(request: Request) -> JSONResponse:
    """Pin the ledger head out-of-band to a file; 404 when off, 503 with no ledger."""
    if not _enabled(request, "enable_audit_anchor"):
        return JSONResponse(
            status_code=404, content={"detail": "audit anchor disabled"}
        )
    ledger = getattr(request.app.state, "hash_audit", None)
    if not isinstance(ledger, HashChainedAudit):
        return JSONResponse(
            status_code=503, content={"detail": "audit ledger unavailable"}
        )
    entries = ledger.entries()
    head = entries[-1].entry_hash if entries else GENESIS_HASH
    anchor = make_anchor(head, now=time.time(), note="api")
    settings = getattr(request.app.state, "settings", None)
    path = Path(getattr(settings, "audit_anchor_path", "data/anchors.jsonl"))

    def _append_anchor(line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    # Off the event loop: a slow/contended disk would otherwise stall every
    # concurrent request for the duration of the append.
    await anyio.to_thread.run_sync(_append_anchor, anchor.model_dump_json() + "\n")
    logger.info("audit anchor written", extra={"head": head[:12]})
    return JSONResponse(status_code=200, content=anchor.model_dump())
