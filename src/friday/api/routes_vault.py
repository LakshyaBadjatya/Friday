"""``/vault`` — the flagged vault REST API (core upload surface, Task 7).

Six surfaces, all gated behind ``FRIDAY_ENABLE_VAULT`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/study`` and
``/perception``):

* ``POST   /vault/sign`` ``{source?, privacy?, space?, device_id?}`` -> creates a
  ``pending`` item and returns a Cloudinary signature scoped to exactly that one
  upload. ``507`` when the owner is over their storage quota.
* ``POST   /vault/items/{id}/commit`` -> verifies the upload actually landed on
  Cloudinary (via its Admin API) before trusting a single field of it; ``409`` if
  Cloudinary does not have the asset.
* ``GET    /vault/items?subject=&kind=&space=&include_locked=&limit=`` -> the
  owner's items.
* ``GET    /vault/items/{id}`` -> one item.
* ``DELETE /vault/items/{id}`` -> deletes the Cloudinary asset, then the index row.
* ``GET    /vault/quota`` -> the owner's storage :class:`~friday.vault.quota.Usage`.

Solver, notes, search, analytics and exam routes are NOT part of this surface —
they are wired in a later task; this module only ever imports the vault's
models/index/cloudinary/quota primitives.

**The upload handshake is the security-critical part.** The phone never holds the
Cloudinary API secret: ``sign`` hands it a signature scoped to one upload, and
``commit`` refuses to believe anything the phone claims about what it uploaded —
size, format, version and the delivery URL are all taken from Cloudinary's OWN
verify response, never from the client. This is what stops a client inventing
items or lying about size to escape the quota. ``CloudinaryAsset.secure_url`` is
populated from that same verify response and served back out of ``GET
/vault/items/{id}`` for display — FRIDAY never mints its own delivery URL (a
hand-rolled one returned 401 against live Cloudinary; see
:mod:`friday.vault.cloudinary`).

The ``public_id`` Cloudinary expects for an item is computed in exactly ONE
place: :meth:`~friday.vault.cloudinary.CloudinaryProvider.upload_params`. Both
``sign`` and ``commit`` read it back off that call's own :class:`UploadPayload`
rather than re-deriving the ``vault/{owner}/{item}`` string locally — a doubled
folder segment from two independent, silently-drifted implementations of that
format string was a real production bug once.

Ownership is enforced implicitly: every index lookup here is scoped to the
caller's own ``owner_uid`` (``(owner_uid, id)`` is the index's primary key), so a
request naming another owner's item id simply misses and 404s — there is no
separate ownership check to forget.

The caller's identity is ``request.state.uid`` when a future auth layer sets it,
falling back to a single local owner so curl and the HUD work without a token
(this is a personal single-owner vault, not a multi-tenant one like ``/circle``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.vault.cloudinary import CloudinaryProvider
from friday.vault.firestore_index import FirestoreIndexError
from friday.vault.index import VaultIndex
from friday.vault.models import CaptureSource, CloudinaryAsset, Item, ItemStatus, Privacy
from friday.vault.quota import QuotaGuard

logger = get_logger("friday.api.routes_vault")

router = APIRouter()

#: The identity used when no caller-identifying middleware has set
#: ``request.state.uid`` — a personal, single-owner vault works fine without one.
_LOCAL_OWNER = "local-owner"


class SignRequest(BaseModel):
    """JSON body for ``POST /vault/sign`` — every field is optional."""

    source: CaptureSource = CaptureSource.CAMERA
    privacy: Privacy = Privacy.PRIVATE
    space: str = "private"
    device_id: str = Field(default="", max_length=200)


def _vault_enabled(request: Request) -> bool:
    """Whether the vault is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_vault", False))


def _disabled() -> JSONResponse:
    """The canonical ``vault disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "vault disabled"})


def _not_found() -> JSONResponse:
    """The canonical ``no such item`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "item not found"})


def _index_unavailable(exc: FirestoreIndexError) -> JSONResponse:
    """Map an index outage to a 503 — never a quiet empty/false result.

    Per :class:`~friday.vault.firestore_index.FirestoreIndexError`'s docstring,
    the route layer (here) owns turning "Firestore could not be trusted" into a
    5xx rather than letting it look like an ordinary empty list, a quota pass, or
    a 404. Every handler below that calls into the index or quota guard catches
    this at its call site and returns this response instead of letting the
    exception become an unhandled 500.
    """
    logger.error("vault index unavailable", extra={"error": str(exc)})
    return JSONResponse(status_code=503, content={"detail": "vault index unavailable"})


def _owner_uid(request: Request) -> str:
    """The caller's uid: ``request.state.uid`` when set, else the local owner."""
    uid = getattr(request.state, "uid", None)
    return uid if isinstance(uid, str) and uid else _LOCAL_OWNER


def _get_index(request: Request) -> VaultIndex:
    """Pull the process-wide vault index off ``app.state``."""
    index = getattr(request.app.state, "vault_index", None)
    if index is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault index is not initialized on app.state")
    return index  # type: ignore[no-any-return]


def _get_cloudinary(request: Request) -> CloudinaryProvider:
    """Pull the process-wide Cloudinary provider off ``app.state``."""
    cloudinary = getattr(request.app.state, "vault_cloudinary", None)
    if cloudinary is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault cloudinary provider is not initialized on app.state")
    return cloudinary  # type: ignore[no-any-return]


def _get_quota(request: Request) -> QuotaGuard:
    """Pull the process-wide quota guard off ``app.state``."""
    quota = getattr(request.app.state, "vault_quota", None)
    if not isinstance(quota, QuotaGuard):  # pragma: no cover - startup guard
        raise RuntimeError("vault quota guard is not initialized on app.state")
    return quota


async def _validate(
    request: Request, model: type[BaseModel]
) -> tuple[BaseModel | None, JSONResponse | None]:
    """Parse + validate the JSON body against ``model``; return (body, error)."""
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return None, JSONResponse(
            status_code=422, content={"detail": "expected a JSON body"}
        )
    try:
        return model.model_validate(raw), None
    except ValidationError as exc:
        return None, JSONResponse(status_code=422, content={"detail": str(exc)})


def _asset_from_verify(public_id: str, verified: dict[str, Any]) -> CloudinaryAsset:
    """Build a :class:`CloudinaryAsset` from Cloudinary's own verify response.

    Every field is taken from ``verified`` (Cloudinary's Admin API answer), never
    from anything the client claimed — that is the entire point of the commit
    step. ``public_id`` falls back to the one we asked Cloudinary about, since a
    healthy response always echoes it back anyway.
    """
    return CloudinaryAsset(
        public_id=str(verified.get("public_id") or public_id),
        version=int(verified.get("version") or 0),
        format=str(verified.get("format") or ""),
        bytes=int(verified.get("bytes") or 0),
        width=int(verified.get("width") or 0),
        height=int(verified.get("height") or 0),
        resource_type=str(verified.get("resource_type") or "image"),
        secure_url=str(verified.get("secure_url") or ""),
    )


@router.post("/vault/sign", response_model=None)
async def sign_upload(request: Request) -> JSONResponse:
    """Create a pending item and sign one Cloudinary upload for it.

    404 when disabled, 422 on a malformed body, 507 when the owner has no room
    left under their storage quota (checked BEFORE a signature is ever issued, so
    an over-quota account cannot start an upload it has no room for).
    """
    if not _vault_enabled(request):
        return _disabled()
    body, error = await _validate(request, SignRequest)
    if error is not None:
        return error
    assert isinstance(body, SignRequest)

    owner_uid = _owner_uid(request)
    quota = _get_quota(request)
    try:
        may_upload = quota.may_upload(owner_uid)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    if not may_upload:
        return JSONResponse(
            status_code=507, content={"detail": "vault storage quota exceeded"}
        )

    cloudinary = _get_cloudinary(request)
    item_id = str(uuid.uuid4())
    payload = cloudinary.upload_params(owner_uid=owner_uid, item_id=item_id)
    item = Item(
        id=item_id,
        owner_uid=owner_uid,
        space=body.space,
        privacy=body.privacy,
        source=body.source,
        status=ItemStatus.PENDING,
        device_id=body.device_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    index = _get_index(request)
    try:
        index.put_item(item)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(
        status_code=200,
        content={"item_id": item.id, "upload": payload.model_dump()},
    )


@router.post("/vault/items/{item_id}/commit", response_model=None)
async def commit_item(request: Request, item_id: str) -> JSONResponse:
    """Verify an upload landed on Cloudinary and record it; refuse if it did not.

    404 when disabled or the item is unknown to (or not owned by) the caller;
    409 when Cloudinary's Admin API does not have the asset — the phone's claim
    that it uploaded something is never trusted on its own. Safe to call twice:
    a second commit just re-verifies and refreshes the recorded asset, and a
    commit on an item that has already progressed past ``uploaded`` (e.g.
    ``ready``, once a later task's solver has run) never regresses its status.
    """
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        item = index.get_item(owner_uid, item_id)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    if item is None:
        return _not_found()

    cloudinary = _get_cloudinary(request)
    public_id = cloudinary.upload_params(owner_uid=owner_uid, item_id=item_id).params[
        "public_id"
    ]
    verified = cloudinary.verify(public_id)
    if verified is None:
        return JSONResponse(
            status_code=409, content={"detail": "asset not found on cloudinary"}
        )

    item.cloudinary = _asset_from_verify(public_id, verified)
    if item.status is ItemStatus.PENDING:
        item.status = ItemStatus.UPLOADED
    try:
        index.put_item(item)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content=item.model_dump(mode="json"))


@router.get("/vault/items", response_model=None)
async def list_items(
    request: Request,
    subject: str = "",
    kind: str = "",
    space: str = "",
    include_locked: bool = False,
    limit: int = 100,
) -> JSONResponse:
    """List the caller's items, optionally filtered. 404 when disabled."""
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        items = index.list_items(
            owner_uid,
            subject=subject,
            kind=kind,
            space=space,
            include_locked=include_locked,
            limit=limit,
        )
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(
        status_code=200,
        content={
            "items": [i.model_dump(mode="json") for i in items],
            "count": len(items),
        },
    )


@router.get("/vault/items/{item_id}", response_model=None)
async def get_item(request: Request, item_id: str) -> JSONResponse:
    """One item, including its ``secure_url`` for display. 404 when disabled or missing."""
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        item = index.get_item(owner_uid, item_id)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    if item is None:
        return _not_found()
    return JSONResponse(status_code=200, content=item.model_dump(mode="json"))


@router.delete("/vault/items/{item_id}", response_model=None)
async def delete_item(request: Request, item_id: str) -> JSONResponse:
    """Delete the Cloudinary asset (if any), then the index row. 404 when unknown."""
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        item = index.get_item(owner_uid, item_id)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    if item is None:
        return _not_found()

    if item.cloudinary is not None:
        cloudinary = _get_cloudinary(request)
        deleted = await cloudinary.delete(item.cloudinary.public_id)
        if not deleted:
            logger.warning(
                "cloudinary delete did not confirm removal; deleting index row anyway",
                extra={"item_id": item_id, "public_id": item.cloudinary.public_id},
            )

    try:
        removed = index.delete_item(owner_uid, item_id)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content={"id": item_id, "removed": removed})


@router.get("/vault/quota", response_model=None)
async def get_quota(request: Request) -> JSONResponse:
    """The caller's storage usage report. 404 when disabled."""
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    quota = _get_quota(request)
    try:
        usage = quota.usage(owner_uid)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content=usage.model_dump())
