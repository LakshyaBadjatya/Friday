"""``/circle`` — REST surface for groups, invites, members, and status.

Flag-gated by ``FRIDAY_ENABLE_CIRCLE`` (every route 404s when off). The caller's
identity is the bearer token resolved via ``app.state.siri_identities`` (token ->
uid); real Firebase ID-token verification replaces that map without changing these
handlers. Services are taken from ``app.state`` when wired (e.g. a Firestore-backed
build), else built lazily on in-memory stores so the surface works out of the box.

Errors map cleanly: an unknown caller -> 401, a guardrail
(:class:`~friday.errors.PermissionError`) -> 403, a bad invite
(:class:`~friday.circle.models.InviteError`) -> 400, an unknown target -> 404.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.circle.models import InviteError
from friday.circle.service import CircleService
from friday.circle.status import InMemoryStatusStore, StatusService
from friday.circle.store import InMemoryCircleStore
from friday.errors import PermissionError
from friday.logging import get_logger

logger = get_logger("friday.api.routes_circle")

router = APIRouter(prefix="/circle")


def _enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_circle", False))


def _disabled() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "circle disabled"})


def _circle(request: Request) -> CircleService:
    service = getattr(request.app.state, "circle", None)
    if not isinstance(service, CircleService):
        service = CircleService(InMemoryCircleStore())
        request.app.state.circle = service
    return service


def _status(request: Request) -> StatusService:
    service = getattr(request.app.state, "circle_status", None)
    if not isinstance(service, StatusService):
        service = StatusService(_circle(request), InMemoryStatusStore())
        request.app.state.circle_status = service
    return service


def _caller_uid(request: Request) -> str | None:
    identities = getattr(request.app.state, "siri_identities", None)
    if not isinstance(identities, dict):
        return None
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    uid = identities.get(auth[7:].strip())
    return uid if isinstance(uid, str) else None


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "unauthorized"})


class CreateGroupBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    display_name: str = Field(default="Me", min_length=1, max_length=200)
    tz: str = "UTC"


class CreateInviteBody(BaseModel):
    email: str | None = Field(default=None, max_length=320)


class AcceptInviteBody(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    tz: str = "UTC"


class SetStatusBody(BaseModel):
    text: str | None = Field(default=None, max_length=2000)
    mood: str | None = Field(default=None, max_length=120)
    place: str | None = Field(default=None, max_length=200)
    arrived_safe: bool | None = None


@router.post("/groups", response_model=None)
async def create_group(request: Request, body: CreateGroupBody) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    group = _circle(request).create_group(
        name=body.name,
        description=body.description,
        admin_uid=uid,
        admin_display_name=body.display_name,
        admin_tz=body.tz,
        now=datetime.now(UTC),
    )
    return JSONResponse(status_code=200, content=group.model_dump(mode="json"))


@router.get("/groups/{group_id}/members", response_model=None)
async def list_members(request: Request, group_id: str) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    members = _circle(request).list_members(group_id)
    if not any(m.uid == uid for m in members):
        return JSONResponse(status_code=403, content={"detail": "not a member"})
    body = [m.model_dump(mode="json") for m in members]
    return JSONResponse(status_code=200, content={"members": body})


@router.post("/groups/{group_id}/invites", response_model=None)
async def create_invite(
    request: Request, group_id: str, body: CreateInviteBody
) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    try:
        invite = _circle(request).invite(
            group_id=group_id, by_uid=uid, email=body.email, now=datetime.now(UTC)
        )
    except PermissionError as exc:
        return JSONResponse(status_code=403, content={"detail": str(exc)})
    return JSONResponse(status_code=200, content=invite.model_dump(mode="json"))


@router.post("/invites/{code}/accept", response_model=None)
async def accept_invite(
    request: Request, code: str, body: AcceptInviteBody
) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    try:
        member = _circle(request).accept_invite(
            code=code,
            uid=uid,
            display_name=body.display_name,
            tz=body.tz,
            now=datetime.now(UTC),
        )
    except InviteError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return JSONResponse(status_code=200, content=member.model_dump(mode="json"))


@router.put("/status", response_model=None)
async def set_status(request: Request, body: SetStatusBody) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    status = _status(request).set_status(
        uid,
        text=body.text,
        mood=body.mood,
        place=body.place,
        arrived_safe=body.arrived_safe,
        now=datetime.now(UTC),
    )
    return JSONResponse(status_code=200, content=status.model_dump(mode="json"))


@router.get("/status/{target_uid}", response_model=None)
async def get_status(request: Request, target_uid: str) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    try:
        spoken = _status(request).describe(uid, target_uid, now=datetime.now(UTC))
    except PermissionError as exc:
        return JSONResponse(status_code=403, content={"detail": str(exc)})
    payload: dict[str, Any] = {"target_uid": target_uid, "speak": spoken}
    return JSONResponse(status_code=200, content=payload)
