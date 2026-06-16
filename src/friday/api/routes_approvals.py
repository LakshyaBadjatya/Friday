# © Lakshya Badjatya — Author
"""``/approvals`` — the out-of-band approval workflow (Wave 2; default off).

Gated behind ``FRIDAY_ENABLE_APPROVALS``; off by default every route 404s. When on,
an :class:`~friday.security.approvals.ApprovalStore` is held on ``app.state`` so an
irreversible action can be raised, surfaced to the owner (e.g. via a phone push —
delivery is the caller's concern), and only proceed once approved.

* ``POST /approvals {action, requested_by?, ttl_seconds?}`` -> the created pending
  request (a server-generated id).
* ``POST /approvals/{id}/approve`` / ``.../deny`` -> the decided request; 404 for an
  unknown id, 409 if it was already decided or has expired.
* ``GET /approvals`` -> the requests still pending now.

Imports no LLM SDK — pure state-machine operations over the shared store.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.security.approvals import ApprovalStore

logger = get_logger("friday.api.routes_approvals")

router = APIRouter()


def _get_store(request: Request) -> ApprovalStore | None:
    """The process-wide :class:`ApprovalStore`, or ``None`` when approvals are off."""
    store = getattr(request.app.state, "approval_store", None)
    return store if isinstance(store, ApprovalStore) else None


class CreateApprovalRequest(BaseModel):
    """Body for ``POST /approvals``."""

    action: str = Field(min_length=1, max_length=2000)
    requested_by: str = Field(default="FRIDAY", max_length=100)
    ttl_seconds: float | None = None


def _disabled() -> JSONResponse:
    """The canonical ``approvals disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "approvals disabled"})


@router.post("/approvals", response_model=None)
async def create_approval(request: Request, body: CreateApprovalRequest) -> JSONResponse:
    """Raise a new pending approval; 404 when the feature is off."""
    store = _get_store(request)
    if store is None:
        return _disabled()
    created = store.create(
        body.action,
        request_id=uuid.uuid4().hex,
        now=time.time(),
        requested_by=body.requested_by,
        ttl_seconds=body.ttl_seconds,
    )
    logger.info("approval requested", extra={"id": created.id})
    return JSONResponse(status_code=200, content=created.model_dump())


@router.post("/approvals/{approval_id}/approve", response_model=None)
async def approve(request: Request, approval_id: str) -> JSONResponse:
    """Approve a pending request; 404 unknown, 409 already-decided/expired."""
    return _decide(request, approval_id, approve=True)


@router.post("/approvals/{approval_id}/deny", response_model=None)
async def deny(request: Request, approval_id: str) -> JSONResponse:
    """Deny a pending request; 404 unknown, 409 already-decided/expired."""
    return _decide(request, approval_id, approve=False)


def _decide(request: Request, approval_id: str, *, approve: bool) -> JSONResponse:
    """Shared approve/deny handler with the store's one-shot guarantees."""
    store = _get_store(request)
    if store is None:
        return _disabled()
    try:
        decided = (
            store.approve(approval_id, now=time.time())
            if approve
            else store.deny(approval_id, now=time.time())
        )
    except KeyError:
        return JSONResponse(status_code=404, content={"detail": "unknown approval"})
    except ValueError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    return JSONResponse(status_code=200, content=decided.model_dump())


@router.get("/approvals", response_model=None)
async def list_pending(request: Request) -> JSONResponse:
    """List the requests still pending now; 404 when the feature is off."""
    store = _get_store(request)
    if store is None:
        return _disabled()
    pending = store.pending(now=time.time())
    return JSONResponse(
        status_code=200, content={"pending": [r.model_dump() for r in pending]}
    )
