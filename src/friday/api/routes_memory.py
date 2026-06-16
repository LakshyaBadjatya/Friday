# © Lakshya Badjatya — Author
"""``/memory`` — knowledge-hygiene surfaces (Wave 3; default off).

Read-only utilities over stored memory, each gated behind its own flag so the
offline build exposes none of them (every route 404s).

* ``POST /memory/contradiction {fact}`` (``FRIDAY_ENABLE_CONTRADICTION``) — check
  whether ``fact`` conflicts with what FRIDAY already believes. It pulls the
  relevant stored facts from the long-term store and runs the contradiction
  detector, returning ``{contradicts, conflicting_source_id, explanation}``. The
  honesty spine made callable: surface a conflict *before* a fact is committed,
  rather than letting memory silently hold two incompatible claims.

Imports no LLM SDK — it drives the injected detector (which depends only on the
``LLMProvider`` contract) and reads the shared long-term store off ``app.state``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.memory.autotag import AutoTagger
from friday.memory.citations import Source
from friday.memory.contradiction import ContradictionDetector

logger = get_logger("friday.api.routes_memory")

router = APIRouter()

# How many candidate stored facts to compare a new fact against.
_CANDIDATE_LIMIT = 8


class ContradictionRequest(BaseModel):
    """Inbound body for ``POST /memory/contradiction`` — the fact to check."""

    fact: str = Field(min_length=1, max_length=4000)


class TagRequest(BaseModel):
    """Inbound body for ``POST /memory/tag`` — the text to tag."""

    text: str = Field(min_length=1, max_length=8000)


def _get_detector(request: Request) -> ContradictionDetector | None:
    """The process-wide :class:`ContradictionDetector`, or ``None`` when off."""
    detector = getattr(request.app.state, "contradiction_detector", None)
    return detector if isinstance(detector, ContradictionDetector) else None


def _candidate_sources(request: Request, fact: str) -> list[Source]:
    """Pull stored facts relevant to ``fact`` from the long-term store as Sources.

    Returns ``[]`` when no long-term store is wired. Each matched fact becomes a
    :class:`~friday.memory.citations.Source` carrying its ``source_id`` and text.
    """
    long_term = getattr(request.app.state, "long_term", None)
    if long_term is None:
        return []
    sources: list[Source] = []
    for stored in long_term.query_facts(fact, limit=_CANDIDATE_LIMIT):
        sources.append(Source(source_id=stored.source_id, text=stored.text))
    return sources


@router.post("/memory/contradiction", response_model=None)
async def check_contradiction(request: Request, body: ContradictionRequest) -> JSONResponse:
    """Check ``fact`` against stored memory; 404 when the feature is off."""
    detector = _get_detector(request)
    if detector is None:
        return JSONResponse(
            status_code=404, content={"detail": "contradiction check disabled"}
        )
    existing = _candidate_sources(request, body.fact)
    result = await detector.check(body.fact, existing)
    logger.info("contradiction check", extra={"candidates": len(existing)})
    return JSONResponse(status_code=200, content=result.model_dump())


def _get_tagger(request: Request) -> AutoTagger | None:
    """The process-wide :class:`AutoTagger`, or ``None`` when auto-tagging is off."""
    tagger = getattr(request.app.state, "autotagger", None)
    return tagger if isinstance(tagger, AutoTagger) else None


@router.post("/memory/tag", response_model=None)
async def suggest_tags(request: Request, body: TagRequest) -> JSONResponse:
    """Suggest normalized topic tags for ``text``; 404 when the feature is off."""
    tagger = _get_tagger(request)
    if tagger is None:
        return JSONResponse(status_code=404, content={"detail": "auto-tagging disabled"})
    tags = await tagger.tag(body.text)
    return JSONResponse(status_code=200, content={"tags": tags})
