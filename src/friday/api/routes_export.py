# © Lakshya Badjatya — Author
"""``/export`` — second-brain Markdown export (Wave 3; default off).

Gated behind ``FRIDAY_ENABLE_KB_EXPORT`` (read off the startup settings on
``app.state``); off by default the route is ``404`` so the feature does not exist
for callers.

* ``GET /export`` -> ``text/markdown`` — the long-term facts rendered as an
  Obsidian-style Markdown note (each fact a bullet annotated with its source id,
  preserving provenance). Reads the shared long-term store off ``app.state`` and
  the pure :mod:`friday.memory.export` renderer; no LLM, no mutation.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from friday.logging import get_logger
from friday.memory.export import facts_to_note, to_markdown

logger = get_logger("friday.api.routes_export")

router = APIRouter()

# Upper bound on facts pulled for one export (newest first).
_EXPORT_LIMIT = 1000


def _export_enabled(request: Request) -> bool:
    """Whether KB export is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_kb_export", False))


@router.get("/export", response_model=None)
async def export_markdown(request: Request) -> JSONResponse | PlainTextResponse:
    """Render the long-term facts as a Markdown note; 404 when the feature is off."""
    if not _export_enabled(request):
        return JSONResponse(status_code=404, content={"detail": "export disabled"})
    long_term = getattr(request.app.state, "long_term", None)
    facts = long_term.all_facts(limit=_EXPORT_LIMIT) if long_term is not None else []
    note = facts_to_note(
        [(fact.text, fact.source_id) for fact in facts],
        title="FRIDAY Knowledge Export",
    )
    logger.info("knowledge export", extra={"facts": len(facts)})
    return PlainTextResponse(to_markdown(note), media_type="text/markdown")
