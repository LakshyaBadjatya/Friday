"""``/graph`` — the flagged knowledge-graph / entity-card API (Tier 2).

Three surfaces, all gated behind ``FRIDAY_ENABLE_KNOWLEDGE_GRAPH`` (read off the
startup settings on ``app.state``); when the flag is off every one is ``404`` so
the feature simply does not exist for callers (mirroring ``/rag`` and
``/reminders``):

* ``GET  /graph/entities?type=`` — list known entities, optionally filtered by
  ``type``; returns ``{entities, count}``.
* ``GET  /graph/entity/{name}`` — the *entity card* for ``name``:
  ``{entity, relations, facts}``, where ``facts`` are the long-term facts that
  mention the name (pulled from the shared long-term store).
* ``POST /graph/extract`` ``{text}`` — run one NON-FATAL LLM extraction pass over
  ``text`` and apply the entities/relations to the graph; returns the applied
  ``{entities, relations}`` (empty when extraction failed — never an error).

The route reads the shared :class:`~friday.graph.store.SQLiteGraphStore` and
:class:`~friday.graph.extractor.EntityExtractor` off ``app.state`` (``app.py``
builds and stashes them when the flag is on) and the shared long-term store for
the card's facts, so an agent-recorded fact and an HTTP entity card see one state.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.graph.extractor import EntityExtractor
from friday.graph.store import SQLiteGraphStore
from friday.logging import get_logger

logger = get_logger("friday.api.routes_graph")

router = APIRouter()


class GraphExtractRequest(BaseModel):
    """JSON body for ``POST /graph/extract``."""

    text: str = Field(min_length=1)


def _graph_enabled(request: Request) -> bool:
    """Whether the knowledge graph is enabled, read off app-state settings."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_knowledge_graph", False))


def _disabled() -> JSONResponse:
    """The canonical ``knowledge graph disabled`` 404 response."""
    return JSONResponse(
        status_code=404, content={"detail": "knowledge graph disabled"}
    )


def _get_store(request: Request) -> SQLiteGraphStore:
    """Pull the process-wide graph store off ``app.state``."""
    store = getattr(request.app.state, "graph_store", None)
    if not isinstance(store, SQLiteGraphStore):  # pragma: no cover - startup guard
        raise RuntimeError("graph store is not initialized on app.state")
    return store


def _get_extractor(request: Request) -> EntityExtractor:
    """Pull the process-wide entity extractor off ``app.state``."""
    extractor = getattr(request.app.state, "graph_extractor", None)
    if not isinstance(extractor, EntityExtractor):  # pragma: no cover - startup guard
        raise RuntimeError("graph extractor is not initialized on app.state")
    return extractor


@router.get("/graph/entities", response_model=None)
async def graph_entities(
    request: Request, type: str | None = None
) -> JSONResponse:
    """List entities (optionally filtered by ``type``); 404 when disabled."""
    if not _graph_enabled(request):
        return _disabled()
    store = _get_store(request)
    entities = store.list_entities(type=type)
    return JSONResponse(
        status_code=200,
        content={
            "entities": [entity.model_dump() for entity in entities],
            "count": len(entities),
        },
    )


@router.get("/graph/entity/{name}", response_model=None)
async def graph_entity_card(request: Request, name: str) -> JSONResponse:
    """Return the entity card for ``name``; 404 when disabled.

    The card aggregates the entity, every relation touching it, and the long-term
    facts that mention the name (from the shared long-term store on ``app.state``,
    when present). An unknown name still returns ``200`` with ``entity: null`` and
    empty relations/facts — the card view never 404s on a missing node, only on a
    disabled feature.
    """
    if not _graph_enabled(request):
        return _disabled()
    store = _get_store(request)
    long_term = getattr(request.app.state, "long_term", None)
    card = store.entity_card(name, long_term=long_term)
    return JSONResponse(status_code=200, content=card)


@router.post("/graph/extract", response_model=None)
async def graph_extract(request: Request) -> JSONResponse:
    """Extract entities/relations from a note and apply them; 404 when disabled.

    JSON body ``{text}`` -> one NON-FATAL LLM pass -> the applied
    ``{entities, relations}``. A bad body is ``422``; an extraction failure is
    *not* an error (it returns an empty ``{entities, relations}`` with ``200``).
    """
    if not _graph_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(
            status_code=422, content={"detail": "expected a JSON body"}
        )
    try:
        body = GraphExtractRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    store = _get_store(request)
    extractor = _get_extractor(request)
    result = await extractor.upsert_into(store, body.text)
    return JSONResponse(status_code=200, content=result)
