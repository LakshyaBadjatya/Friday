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

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from friday.circle.auth import FirebaseTokenVerifier, TokenVerifier, resolve_caller
from friday.circle.chat import (
    ChatBroadcaster,
    ChatService,
    ChatStore,
    InMemoryChatStore,
    StreamTicketStore,
)
from friday.circle.firebase import FirebaseBackend, get_backend
from friday.circle.firestore_store import FirestoreChatStore, FirestoreCircleStore
from friday.circle.models import InviteError
from friday.circle.service import CircleService
from friday.circle.status import InMemoryStatusStore, StatusService
from friday.circle.store import CircleStore, InMemoryCircleStore
from friday.errors import PermissionError
from friday.logging import get_logger

logger = get_logger("friday.api.routes_circle")

router = APIRouter(prefix="/circle")


def _enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_circle", False))


def _disabled() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "circle disabled"})


def _backend(request: Request) -> FirebaseBackend | None:
    """The lazily-built Firebase backend (None when no service account is set)."""
    state = request.app.state
    if getattr(state, "_circle_backend_checked", False):
        backend = getattr(state, "_circle_backend", None)
        return backend if isinstance(backend, FirebaseBackend) else None
    settings = getattr(state, "settings", None)
    secret = getattr(settings, "firebase_service_account", None)
    raw = secret.get_secret_value() if secret is not None else None
    project = getattr(settings, "firebase_project_id", "") or ""
    backend = get_backend(raw, project)
    state._circle_backend = backend
    state._circle_backend_checked = True
    return backend


def _verifier(request: Request) -> TokenVerifier | None:
    """A real Firebase ID-token verifier when a backend exists, else None."""
    state = request.app.state
    if getattr(state, "_circle_verifier_checked", False):
        existing = getattr(state, "_circle_verifier", None)
        return existing if isinstance(existing, FirebaseTokenVerifier) else None
    backend = _backend(request)
    verifier = FirebaseTokenVerifier(backend.app) if backend is not None else None
    state._circle_verifier = verifier
    state._circle_verifier_checked = True
    return verifier


def _circle(request: Request) -> CircleService:
    service = getattr(request.app.state, "circle", None)
    if isinstance(service, CircleService):
        return service
    backend = _backend(request)
    store: CircleStore = (
        FirestoreCircleStore(backend.firestore)
        if backend is not None
        else InMemoryCircleStore()
    )
    service = CircleService(store)
    request.app.state.circle = service
    return service


def _broadcaster(request: Request) -> ChatBroadcaster:
    bc = getattr(request.app.state, "circle_broadcaster", None)
    if not isinstance(bc, ChatBroadcaster):
        bc = ChatBroadcaster()
        request.app.state.circle_broadcaster = bc
    return bc


def _chat(request: Request) -> ChatService:
    service = getattr(request.app.state, "circle_chat", None)
    if isinstance(service, ChatService):
        return service
    backend = _backend(request)
    store: ChatStore = (
        FirestoreChatStore(backend.firestore)
        if backend is not None
        else InMemoryChatStore()
    )
    service = ChatService(_circle(request), store, _broadcaster(request))
    request.app.state.circle_chat = service
    return service


def _status(request: Request) -> StatusService:
    service = getattr(request.app.state, "circle_status", None)
    if not isinstance(service, StatusService):
        service = StatusService(_circle(request), InMemoryStatusStore())
        request.app.state.circle_status = service
    return service


def _tickets(request: Request) -> StreamTicketStore:
    store = getattr(request.app.state, "circle_stream_tickets", None)
    if not isinstance(store, StreamTicketStore):
        store = StreamTicketStore()
        request.app.state.circle_stream_tickets = store
    return store


def _caller_uid(request: Request) -> str | None:
    """Resolve the ``Authorization`` bearer token to a uid.

    A real Firebase ID token is verified via ``firebase-admin``; the dev/Siri
    token->uid map on app state is the fallback, so both resolve through one path.
    """
    identities = getattr(request.app.state, "siri_identities", None)
    identity_map = identities if isinstance(identities, dict) else None
    header = request.headers.get("authorization")
    return resolve_caller(header, verifier=_verifier(request), identities=identity_map)


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


class PostMessageBody(BaseModel):
    # Base64 AES-GCM ciphertext + nonce, sealed in the browser. The server never
    # sees plaintext or the key (E2EE) — it only relays and stores these two.
    ciphertext: str = Field(min_length=1, max_length=20000)
    nonce: str = Field(min_length=1, max_length=64)


@router.get("/groups", response_model=None)
async def my_groups(request: Request) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    groups = _circle(request).groups_for(uid)
    body = [g.model_dump(mode="json") for g in groups]
    return JSONResponse(status_code=200, content={"groups": body})


@router.get("/invites/{code}", response_model=None)
async def preview_invite(request: Request, code: str) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    group = _circle(request).peek_invite(code)
    if group is None:
        return JSONResponse(status_code=404, content={"detail": "invite not found"})
    return JSONResponse(
        status_code=200, content={"group": {"id": group.id, "name": group.name}}
    )


@router.post("/groups/{group_id}/messages", response_model=None)
async def post_message(
    request: Request, group_id: str, body: PostMessageBody
) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    try:
        message = _chat(request).post(
            group_id=group_id,
            sender_uid=uid,
            ciphertext=body.ciphertext,
            nonce=body.nonce,
            now=datetime.now(UTC),
        )
    except PermissionError as exc:
        return JSONResponse(status_code=403, content={"detail": str(exc)})
    return JSONResponse(status_code=200, content=message.model_dump(mode="json"))


@router.get("/groups/{group_id}/messages", response_model=None)
async def list_messages(request: Request, group_id: str) -> JSONResponse:
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    try:
        messages = _chat(request).history(group_id=group_id, requester_uid=uid)
    except PermissionError as exc:
        return JSONResponse(status_code=403, content={"detail": str(exc)})
    body = [m.model_dump(mode="json") for m in messages]
    return JSONResponse(status_code=200, content={"messages": body})


@router.post("/groups/{group_id}/stream/ticket", response_model=None)
async def stream_ticket(request: Request, group_id: str) -> JSONResponse:
    """Mint a single-use, short-lived ticket to open the SSE stream.

    Authenticated by the normal ``Authorization`` header (so the bearer token never
    rides in a URL). The returned opaque ticket is bound to (uid, group_id), expires
    in ~60s, and is consumed on first use.
    """
    if not _enabled(request):
        return _disabled()
    uid = _caller_uid(request)
    if uid is None:
        return _unauthorized()
    members = _circle(request).list_members(group_id)
    if not any(m.uid == uid for m in members):
        return JSONResponse(status_code=403, content={"detail": "not a member"})
    ticket = _tickets(request).mint(
        uid=uid, group_id=group_id, now=datetime.now(UTC)
    )
    return JSONResponse(status_code=200, content={"ticket": ticket})


@router.get("/groups/{group_id}/stream", response_model=None)
async def stream_messages(
    request: Request, group_id: str, ticket: str | None = None
) -> Response:
    """Server-Sent-Events stream of new ciphertext for a member of the group.

    Authorized by a single-use ``ticket`` (minted via the POST above), NOT a bearer
    token — so nothing sensitive appears in the URL or access logs. Each event's
    ``data`` is a JSON ChatMessage (ciphertext only); the client decrypts locally.
    """
    if not _enabled(request):
        return _disabled()
    if not ticket:
        return _unauthorized()
    entry = _tickets(request).consume(ticket, now=datetime.now(UTC))
    if entry is None or entry.group_id != group_id:
        return _unauthorized()
    uid = entry.uid
    members = _circle(request).list_members(group_id)
    if not any(m.uid == uid for m in members):
        return JSONResponse(status_code=403, content={"detail": "not a member"})
    broadcaster = _chat(request).broadcaster
    queue = broadcaster.subscribe(group_id)

    async def events() -> Any:
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                payload = json.dumps(message.model_dump(mode="json"))
                yield f"data: {payload}\n\n"
        finally:
            broadcaster.unsubscribe(group_id, queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
