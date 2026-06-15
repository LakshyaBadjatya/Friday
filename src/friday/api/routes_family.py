"""``/family`` — the flagged consent-enforced family-sharing API (build-spec §18).

Five surfaces, all gated behind ``FRIDAY_ENABLE_FAMILY_SHARING`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off every
one of them is ``404`` so the feature simply does not exist for callers
(mirroring ``/maps`` and ``/hud``):

* ``POST /family/optin``   ``{name, self_opt_in=true}`` -> the opted-in member.
  Guardrail 1: ``self_opt_in`` MUST be true — a participant can only be added by
  THEMSELVES; an attempt to add someone from another account is ``403``.
* ``POST /family/share``   ``{owner, viewer, raw_location=false}`` -> the updated
  member. Only an opted-in owner may share (else ``403``). The default share is
  the coarse geofence status; ``raw_location`` records an explicit per-viewer
  raw-coordinate grant.
* ``POST /family/revoke``  ``{owner, viewer}`` -> the updated member. Guardrail
  3: the revoke stops sharing INSTANTLY (a later view -> ``403``).
* ``GET  /family/status/{name}?viewer=`` -> the shared location AND RECORDS the
  view. Default is geofence STATUS, not coordinates; raw only on an explicit
  per-viewer grant. ``403`` when the owner is not sharing with the viewer.
* ``GET  /family/views/{name}`` -> ``{views, count}`` — who viewed ``name``
  (guardrail 4: the viewed member can see who viewed them).

The store is built lazily inside the route from ``get_settings().memory_db_path``
(a sibling SQLite database, reusing the existing path setting) and cached for the
life of the process; tests reset it via :func:`reset_store`. The consent
guardrails are enforced by the :class:`~friday.family.service.FamilyService`, so
the route stays thin.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.config import get_settings
from friday.family.service import FamilyService, FamilyShareError
from friday.family.store import SQLiteFamilyStore
from friday.logging import get_logger

logger = get_logger("friday.api.routes_family")

router = APIRouter()

#: Process-wide lazily-built family store, keyed by the db path it was built for.
_STORE: SQLiteFamilyStore | None = None
_STORE_PATH: str | None = None


class OptInRequest(BaseModel):
    """JSON body for ``POST /family/optin``.

    ``self_opt_in`` defaults to ``True`` (a member opting THEMSELVES in); a
    caller that explicitly sets it ``False`` is attempting to add someone from
    another account, which the service rejects (guardrail 1).
    """

    name: str = Field(min_length=1, max_length=200)
    self_opt_in: bool = True


class ShareRequest(BaseModel):
    """JSON body for ``POST /family/share`` (and ``/family/revoke`` ignores raw)."""

    owner: str = Field(min_length=1, max_length=200)
    viewer: str = Field(min_length=1, max_length=200)
    raw_location: bool = False


class RevokeRequest(BaseModel):
    """JSON body for ``POST /family/revoke``."""

    owner: str = Field(min_length=1, max_length=200)
    viewer: str = Field(min_length=1, max_length=200)


def _family_enabled() -> bool:
    """Whether family sharing is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_family_sharing", False))


def _disabled() -> JSONResponse:
    """The canonical ``family disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "family sharing disabled"})


def _forbidden(message: str) -> JSONResponse:
    """A consent-guardrail rejection (403) — never leaks anything beyond ``message``."""
    return JSONResponse(status_code=403, content={"detail": message})


def reset_store() -> None:
    """Drop the cached process-wide store (used by tests for isolation)."""
    global _STORE, _STORE_PATH
    _STORE = None
    _STORE_PATH = None


def _get_service() -> FamilyService:
    """Build (once, lazily) and return the family service over the configured db.

    The store is keyed by ``memory_db_path``; if the configured path changes
    (e.g. a test swaps settings) the store is rebuilt for the new path.
    """
    global _STORE, _STORE_PATH
    path = get_settings().memory_db_path
    if _STORE is None or _STORE_PATH != path:
        _STORE = SQLiteFamilyStore(path)
        _STORE_PATH = path
    return FamilyService(_STORE)


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


@router.post("/family/optin", response_model=None)
async def family_optin(request: Request) -> JSONResponse:
    """Opt a member in (self-opt-in); 404 when disabled, 422 bad body, 403 non-self."""
    if not _family_enabled():
        return _disabled()
    body, error = await _validate(request, OptInRequest)
    if error is not None:
        return error
    assert isinstance(body, OptInRequest)
    service = _get_service()
    try:
        participant = service.opt_in(body.name, self_opt_in=body.self_opt_in)
    except FamilyShareError as exc:
        return _forbidden(str(exc))
    return JSONResponse(status_code=200, content=participant.model_dump())


@router.post("/family/share", response_model=None)
async def family_share(request: Request) -> JSONResponse:
    """Share a member's location with a viewer; 404 disabled, 422 bad body, 403 not-opted-in."""
    if not _family_enabled():
        return _disabled()
    body, error = await _validate(request, ShareRequest)
    if error is not None:
        return error
    assert isinstance(body, ShareRequest)
    service = _get_service()
    try:
        participant = service.share(
            body.owner, body.viewer, raw_location=body.raw_location
        )
    except FamilyShareError as exc:
        return _forbidden(str(exc))
    return JSONResponse(status_code=200, content=participant.model_dump())


@router.post("/family/revoke", response_model=None)
async def family_revoke(request: Request) -> JSONResponse:
    """Unilaterally revoke a share (stops sharing instantly); 404/422/403 as above."""
    if not _family_enabled():
        return _disabled()
    body, error = await _validate(request, RevokeRequest)
    if error is not None:
        return error
    assert isinstance(body, RevokeRequest)
    service = _get_service()
    try:
        participant = service.revoke(body.owner, body.viewer)
    except FamilyShareError as exc:
        return _forbidden(str(exc))
    return JSONResponse(status_code=200, content=participant.model_dump())


@router.get("/family/status/{name}", response_model=None)
async def family_status(request: Request, name: str, viewer: str) -> JSONResponse:
    """Return a member's shared location for ``viewer`` and RECORD the view.

    Default granularity is the geofence status (``precision="status"``); raw
    coordinates only on an explicit per-viewer grant. ``403`` when ``name`` is
    not sharing with ``viewer`` (and no view is recorded for the denied attempt).
    """
    if not _family_enabled():
        return _disabled()
    service = _get_service()
    try:
        view = service.view(name, viewer=viewer)
    except FamilyShareError as exc:
        return _forbidden(str(exc))
    return JSONResponse(status_code=200, content=view)


@router.get("/family/views/{name}", response_model=None)
async def family_views(request: Request, name: str) -> JSONResponse:
    """Return who viewed ``name`` (most-recent first); 404 when disabled.

    Guardrail 4: the viewed member can always see who viewed them.
    """
    if not _family_enabled():
        return _disabled()
    service = _get_service()
    views = service.views_of(name)
    return JSONResponse(
        status_code=200, content={"views": views, "count": len(views)}
    )
