"""``/vault`` — the flagged vault REST API (Task 7: upload surface + Task 8:
solver/notes/search/analytics/exam).

All surfaces are gated behind ``FRIDAY_ENABLE_VAULT`` (read off the startup
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
* ``POST   /vault/items/{id}/solve`` -> re-runs :class:`~friday.vault.solver.Solver`
  over one item's extracted text and persists the resulting
  :class:`~friday.vault.models.Solve`. ``403`` when the item is locked — a
  locked capture's text must never reach a model provider, so this is checked
  before the solver is ever called, not left to the solver to enforce.
* ``POST   /vault/notes`` ``{item_ids, prompt?}`` -> writes a
  :class:`~friday.vault.models.Note` over the given items via
  :class:`~friday.vault.notes.NoteWriter` (which itself drops locked/missing
  sources before any model call — see that module).
* ``GET    /vault/notes`` / ``GET /vault/notes/{id}`` -> the owner's notes.
* ``GET    /vault/search?q=&limit=`` -> :class:`~friday.vault.search.VaultSearch`
  hits (locked items are excluded by construction — see that module).
* ``GET    /vault/analytics`` -> the owner's chapter mastery rollup from
  :class:`~friday.vault.analytics.Analytics` (also excludes locked items).
* ``POST   /vault/exam/start`` ``{paper_item_ids, duration_s?}`` -> opens a timed
  :class:`~friday.vault.exam.ExamRunner` session.
* ``POST   /vault/exam/{session_id}/grade`` ``{answer_item_ids}`` -> marks the
  session. ``404`` (never ``403``) when the session is unknown or belongs to
  another owner — :meth:`~friday.vault.exam.ExamRunner.grade` raises
  ``KeyError`` for both cases deliberately, so a caller cannot use the status
  code to probe for another owner's session ids; this route preserves that by
  mapping both to the same 404.

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

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.errors import ProviderError
from friday.logging import get_logger
from friday.vault.analytics import Analytics
from friday.vault.cloudinary import CloudinaryProvider
from friday.vault.exam import ExamRunner
from friday.vault.firestore_index import FirestoreIndexError
from friday.vault.index import VaultIndex
from friday.vault.models import CaptureSource, CloudinaryAsset, Item, ItemStatus, Privacy
from friday.vault.notes import NoteWriter
from friday.vault.quota import QuotaGuard
from friday.vault.search import VaultSearch
from friday.vault.solver import Solver

logger = get_logger("friday.api.routes_vault")

#: Strong references to in-flight pipeline tasks. asyncio only holds a weak
#: reference to a running task, so without this a capture's processing can be
#: collected mid-chain and simply never finish.
_BACKGROUND: set[asyncio.Task[None]] = set()

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


class NoteRequest(BaseModel):
    """JSON body for ``POST /vault/notes`` — ``item_ids`` is required."""

    item_ids: list[str]
    prompt: str = ""


class ExamStartRequest(BaseModel):
    """JSON body for ``POST /vault/exam/start`` — ``paper_item_ids`` is required."""

    paper_item_ids: list[str]
    duration_s: int = 0


class ExamGradeRequest(BaseModel):
    """JSON body for ``POST /vault/exam/{session_id}/grade``."""

    answer_item_ids: list[str]


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


def _get_solver(request: Request) -> Solver:
    """Pull the process-wide :class:`Solver` off ``app.state``."""
    solver = getattr(request.app.state, "vault_solver", None)
    if solver is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault solver is not initialized on app.state")
    return solver  # type: ignore[no-any-return]


def _get_notes(request: Request) -> NoteWriter:
    """Pull the process-wide :class:`NoteWriter` off ``app.state``."""
    notes = getattr(request.app.state, "vault_notes", None)
    if notes is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault note writer is not initialized on app.state")
    return notes  # type: ignore[no-any-return]


def _get_search(request: Request) -> VaultSearch:
    """Pull the process-wide :class:`VaultSearch` off ``app.state``."""
    search = getattr(request.app.state, "vault_search", None)
    if search is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault search is not initialized on app.state")
    return search  # type: ignore[no-any-return]


def _get_exam(request: Request) -> ExamRunner:
    """Pull the process-wide :class:`ExamRunner` off ``app.state``."""
    exam = getattr(request.app.state, "vault_exam", None)
    if exam is None:  # pragma: no cover - startup guard
        raise RuntimeError("vault exam runner is not initialized on app.state")
    return exam  # type: ignore[no-any-return]


def _provider_error(exc: ProviderError) -> JSONResponse:
    """Map a model-provider failure to a clean 502 rather than a 500 traceback."""
    logger.warning("vault provider error", extra={"error": str(exc)})
    return JSONResponse(status_code=502, content={"detail": str(exc)})


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

    # Hand the capture to the pipeline: OCR, classify, maybe solve. Fire and
    # forget, because the phone is holding a spinner and this chain runs a model
    # or two; the item's own status carries the progress instead. Without this a
    # committed capture stops at UPLOADED forever — the photograph is stored and
    # never read, which is the whole feature failing quietly.
    pipeline = getattr(request.app.state, "vault_pipeline", None)
    if pipeline is not None:
        task = asyncio.create_task(pipeline.process(owner_uid, item.id))
        # Hold a reference until it finishes: a bare create_task may be
        # garbage-collected mid-flight, losing the processing silently.
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)

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


@router.post("/vault/items/{item_id}/solve", response_model=None)
async def solve_item(request: Request, item_id: str) -> JSONResponse:
    """Re-run the always-full ensemble on one item and persist the result.

    404 when disabled or the item is unknown to (or not owned by) the caller.
    403 when the item is :attr:`~friday.vault.models.Privacy.LOCKED`
    (``may_leave_for_model()`` is ``False``) — checked here, before the solver
    is ever invoked, so a locked capture's extracted text never reaches a
    model provider. The solver itself never raises on a provider failure (a
    dead operator just drops out of the panel — see
    :mod:`friday.vault.solver`), so no provider-error handling is needed here.
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
    if not item.may_leave_for_model():
        return JSONResponse(status_code=403, content={"detail": "item is locked"})

    solver = _get_solver(request)
    ocr_text = item.ocr_text.strip() or item.caption.strip()
    solve = await solver.solve(item_ids=[item_id], ocr_text=ocr_text)

    item.solve_id = solve.id
    try:
        index.put_solve(solve)
        index.put_item(item)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content=solve.model_dump(mode="json"))


@router.post("/vault/notes", response_model=None)
async def create_note(request: Request) -> JSONResponse:
    """Write a note over ``item_ids``. 404 when disabled, 422 without ``item_ids``.

    Missing or locked source items are silently dropped by
    :class:`~friday.vault.notes.NoteWriter` before any model call; when none
    survive, a minimal (empty) note is still created and persisted rather than
    the call failing — see that module's docstring. A provider failure inside
    note generation also never raises (it degrades to a bare-bones note), so
    no provider-error handling is needed here.
    """
    if not _vault_enabled(request):
        return _disabled()
    body, error = await _validate(request, NoteRequest)
    if error is not None:
        return error
    assert isinstance(body, NoteRequest)

    owner_uid = _owner_uid(request)
    notes = _get_notes(request)
    try:
        note = await notes.write(owner_uid, body.item_ids, prompt=body.prompt)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content=note.model_dump(mode="json"))


@router.get("/vault/notes", response_model=None)
async def list_notes(request: Request, limit: int = 100) -> JSONResponse:
    """List the caller's notes. 404 when disabled."""
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        notes = index.list_notes(owner_uid, limit=limit)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(
        status_code=200,
        content={"notes": [n.model_dump(mode="json") for n in notes], "count": len(notes)},
    )


@router.get("/vault/notes/{note_id}", response_model=None)
async def get_note(request: Request, note_id: str) -> JSONResponse:
    """One note. 404 when disabled, unknown, or owned by another caller.

    ``VaultIndex.get_note`` is scoped to ``(owner_uid, note_id)`` — the same
    ownership-by-query guarantee the item routes rely on — so a note id that
    exists but belongs to a different owner simply misses and 404s.
    """
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        note = index.get_note(owner_uid, note_id)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    if note is None:
        return _not_found()
    return JSONResponse(status_code=200, content=note.model_dump(mode="json"))


@router.get("/vault/search", response_model=None)
async def search_vault(request: Request, q: str = "", limit: int = 20) -> JSONResponse:
    """Search the caller's vault. 404 when disabled.

    Locked items never appear in the results — see
    :mod:`friday.vault.search` for why that guarantee holds even when a
    vector store is configured.
    """
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    search = _get_search(request)
    try:
        hits = await search.search(owner_uid, q, limit=limit)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(
        status_code=200,
        content={"hits": [h.model_dump() for h in hits], "count": len(hits)},
    )


@router.get("/vault/analytics", response_model=None)
async def get_analytics(request: Request) -> JSONResponse:
    """The caller's per-chapter mastery rollup. 404 when disabled.

    Built fresh over the shared index on every call — :class:`Analytics` is a
    thin, stateless wrapper with no other collaborators, so it needs no seam
    on ``app.state``. Locked items are excluded — see
    :mod:`friday.vault.analytics`.
    """
    if not _vault_enabled(request):
        return _disabled()
    owner_uid = _owner_uid(request)
    index = _get_index(request)
    try:
        rows = Analytics(index).rollup(owner_uid)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(
        status_code=200,
        content={"chapters": [r.model_dump() for r in rows], "count": len(rows)},
    )


@router.post("/vault/exam/start", response_model=None)
async def start_exam(request: Request) -> JSONResponse:
    """Open a timed exam session over an already-photographed paper.

    404 when disabled, 422 without ``paper_item_ids``. Sessions live in the
    :class:`~friday.vault.exam.ExamRunner`'s own in-process dict, not the
    index — see that module's docstring for why that is deliberate — so no
    ``FirestoreIndexError`` handling applies here.
    """
    if not _vault_enabled(request):
        return _disabled()
    body, error = await _validate(request, ExamStartRequest)
    if error is not None:
        return error
    assert isinstance(body, ExamStartRequest)

    owner_uid = _owner_uid(request)
    exam = _get_exam(request)
    session = exam.start(owner_uid, body.paper_item_ids, duration_s=body.duration_s)
    return JSONResponse(status_code=200, content=session.model_dump(mode="json"))


@router.post("/vault/exam/{session_id}/grade", response_model=None)
async def grade_exam(request: Request, session_id: str) -> JSONResponse:
    """Mark the answer script against the paper. 404 when disabled or malformed body 422s.

    404 (never 403) when ``session_id`` is unknown or belongs to a different
    owner: :meth:`~friday.vault.exam.ExamRunner.grade` raises ``KeyError`` for
    both cases on purpose, so a caller cannot use the status code to probe for
    another owner's session ids — this route preserves that by mapping both
    to the same 404 rather than distinguishing them. 422 when nothing in the
    paper/answer items resolves to gradable text (every item missing, locked,
    or textless). 502 on a model-provider failure — the one case
    :meth:`ExamRunner.grade` deliberately lets propagate rather than
    swallowing, so it is mapped here instead of leaking a 500.
    """
    if not _vault_enabled(request):
        return _disabled()
    body, error = await _validate(request, ExamGradeRequest)
    if error is not None:
        return error
    assert isinstance(body, ExamGradeRequest)

    owner_uid = _owner_uid(request)
    exam = _get_exam(request)
    try:
        session = await exam.grade(owner_uid, session_id, body.answer_item_ids)
    except KeyError:
        return JSONResponse(status_code=404, content={"detail": "exam session not found"})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    except ProviderError as exc:
        return _provider_error(exc)
    except FirestoreIndexError as exc:
        return _index_unavailable(exc)
    return JSONResponse(status_code=200, content=session.model_dump(mode="json"))
