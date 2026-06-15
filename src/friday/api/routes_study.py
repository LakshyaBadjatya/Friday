"""``/study`` — the flagged study / productivity REST API (Tier 2).

Seven surfaces, all gated behind ``FRIDAY_ENABLE_STUDY`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/reminders`` and
``/meetings``):

* ``POST   /study/cards`` ``{deck, front, back}`` -> the created flashcard.
* ``GET    /study/cards?deck=`` -> ``{cards, count}`` (optionally filtered).
* ``GET    /study/review`` -> ``{cards, count}`` — the cards due for utcnow.
* ``POST   /study/review/{id}`` ``{grade}`` -> the advanced flashcard (404 if no
  such card; 422 on an out-of-range grade).
* ``DELETE /study/cards/{id}`` -> ``{id, removed}`` (idempotent).
* ``POST   /study/sessions`` ``{topic, minutes}`` -> the logged session.
* ``GET    /study/sessions?limit=`` -> ``{sessions, count}`` (most-recent first).

The route reads the shared :class:`~friday.study.store.SQLiteStudyStore` off
``app.state.study_store`` (``app.py`` builds and stashes it when the flag is on).
``GET /study/review`` reads "now" from the wall clock (``utcnow``) at request time
— the *store* keeps its clock injected (the tested store paths drive it), so this
route's single wall-clock read is the only un-injected timing and is not unit
tested for timing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.study.store import SQLiteStudyStore

logger = get_logger("friday.api.routes_study")

router = APIRouter()


class CreateCardRequest(BaseModel):
    """JSON body for ``POST /study/cards``."""

    deck: str = Field(min_length=1, max_length=200)
    front: str = Field(min_length=1, max_length=4000)
    back: str = Field(min_length=1, max_length=4000)


class ReviewRequest(BaseModel):
    """JSON body for ``POST /study/review/{id}`` (grade is the SM-2 recall 0..5)."""

    grade: int = Field(ge=0, le=5)


class SessionRequest(BaseModel):
    """JSON body for ``POST /study/sessions``."""

    topic: str = Field(min_length=1, max_length=500)
    minutes: int = Field(ge=0, le=100_000)


def _study_enabled(request: Request) -> bool:
    """Whether study is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_study", False))


def _disabled() -> JSONResponse:
    """The canonical ``study disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "study disabled"})


def _get_store(request: Request) -> SQLiteStudyStore:
    """Pull the process-wide study store off ``app.state``."""
    store = getattr(request.app.state, "study_store", None)
    if not isinstance(store, SQLiteStudyStore):  # pragma: no cover - startup guard
        raise RuntimeError("study store is not initialized on app.state")
    return store


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


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
@router.post("/study/cards", response_model=None)
async def create_card(request: Request) -> JSONResponse:
    """Create a flashcard from a JSON body; 404 when disabled, 422 on bad body."""
    if not _study_enabled(request):
        return _disabled()
    body, error = await _validate(request, CreateCardRequest)
    if error is not None:
        return error
    assert isinstance(body, CreateCardRequest)
    store = _get_store(request)
    card = store.add_card(body.deck, body.front, body.back)
    return JSONResponse(status_code=200, content=card.model_dump())


@router.get("/study/cards", response_model=None)
async def list_cards(request: Request, deck: str | None = None) -> JSONResponse:
    """List flashcards (optionally filtered by ``deck``); 404 when disabled."""
    if not _study_enabled(request):
        return _disabled()
    store = _get_store(request)
    cards = store.list_cards(deck=deck)
    return JSONResponse(
        status_code=200,
        content={"cards": [c.model_dump() for c in cards], "count": len(cards)},
    )


@router.get("/study/review", response_model=None)
async def review_due(request: Request) -> JSONResponse:
    """List the cards due for utcnow; 404 when disabled.

    "Now" is read from the wall clock at request time; the store's own clock
    stays injected (the tested store paths drive scheduling deterministically).
    """
    if not _study_enabled(request):
        return _disabled()
    store = _get_store(request)
    cards = store.due_cards(datetime.now(UTC))
    return JSONResponse(
        status_code=200,
        content={"cards": [c.model_dump() for c in cards], "count": len(cards)},
    )


@router.post("/study/review/{card_id}", response_model=None)
async def review_card(request: Request, card_id: int) -> JSONResponse:
    """Apply SM-2 for a card's recall grade; 404 when disabled or no such card."""
    if not _study_enabled(request):
        return _disabled()
    body, error = await _validate(request, ReviewRequest)
    if error is not None:
        return error
    assert isinstance(body, ReviewRequest)
    store = _get_store(request)
    card = store.review_card(card_id, body.grade)
    if card is None:
        return JSONResponse(
            status_code=404, content={"detail": f"no card with id {card_id}"}
        )
    return JSONResponse(status_code=200, content=card.model_dump())


@router.delete("/study/cards/{card_id}", response_model=None)
async def delete_card(request: Request, card_id: int) -> JSONResponse:
    """Delete a flashcard by id; 404 when disabled. Idempotent (``removed`` 0/1)."""
    if not _study_enabled(request):
        return _disabled()
    store = _get_store(request)
    removed = store.delete_card(card_id)
    return JSONResponse(
        status_code=200, content={"id": card_id, "removed": removed}
    )


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@router.post("/study/sessions", response_model=None)
async def create_session(request: Request) -> JSONResponse:
    """Log a study session from a JSON body; 404 when disabled, 422 on bad body."""
    if not _study_enabled(request):
        return _disabled()
    body, error = await _validate(request, SessionRequest)
    if error is not None:
        return error
    assert isinstance(body, SessionRequest)
    store = _get_store(request)
    session = store.add_session(body.topic, body.minutes)
    return JSONResponse(status_code=200, content=session.model_dump())


@router.get("/study/sessions", response_model=None)
async def list_sessions(request: Request, limit: int = 20) -> JSONResponse:
    """List logged study sessions, most-recent first; 404 when disabled."""
    if not _study_enabled(request):
        return _disabled()
    store = _get_store(request)
    sessions = store.list_sessions(limit=limit)
    return JSONResponse(
        status_code=200,
        content={
            "sessions": [s.model_dump() for s in sessions],
            "count": len(sessions),
        },
    )
