"""``/rag`` — the personal-RAG ingestion API (Tier 1, Stage 1A).

Three surfaces, all gated behind ``FRIDAY_ENABLE_RAG`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/studio`` and ``/voice``):

* ``POST   /rag/ingest`` — accepts either a JSON ``{source_id, text}`` body *or*
  a ``multipart/form-data`` file upload (the source id is derived from the
  uploaded filename). The document is chunked into the shared vector store and a
  single listable/forgettable marker fact is recorded; returns
  ``{source_id, chunks}``.
* ``GET    /rag/sources`` — lists the ingested source ids (read from the
  long-term marker facts).
* ``DELETE /rag/sources/{source_id}`` — forgets the document from both stores.

Because the ingested chunks live in the *same* vector store the unchanged
:class:`~friday.agents.knowledge.KnowledgeAgent` retrieves from, an ingested note
is immediately answerable (with citations) through the normal ``/chat`` path — no
agent change is needed here.

Multipart parsing is done with a small, dependency-free parser in this module:
``python-multipart`` (which Starlette's form parsing needs) is intentionally not
a project dependency, so the upload body is split on its boundary by hand. The
parser only extracts the first file part's filename + bytes, which is all this
route needs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.rag.ingest import DocumentIngestor

logger = get_logger("friday.api.routes_rag")

router = APIRouter()

# The marker text prefix written per ingested document; ``GET /rag/sources``
# strips it back off the long-term fact text to recover the source ids. Kept in
# sync with :data:`friday.rag.ingest._INGESTED_PREFIX`.
_INGESTED_PREFIX = "ingested "

# Upper bound on marker facts scanned when listing sources — generous, but keeps
# the listing bounded regardless of how many documents were ingested.
_SOURCES_LIMIT = 1000


class RagIngestRequest(BaseModel):
    """JSON body for ``POST /rag/ingest`` (the non-upload path)."""

    source_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)


def _rag_enabled(request: Request) -> bool:
    """Whether RAG is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_rag", False))


def _disabled() -> JSONResponse:
    """The canonical ``rag disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "rag disabled"})


def _get_ingestor(request: Request) -> DocumentIngestor:
    """Pull the process-wide :class:`DocumentIngestor` off ``app.state``."""
    ingestor = getattr(request.app.state, "rag_ingestor", None)
    if not isinstance(ingestor, DocumentIngestor):  # pragma: no cover - startup guard
        raise RuntimeError("rag ingestor is not initialized on app.state")
    return ingestor


def _source_id_from_filename(filename: str) -> str:
    """Derive a source id from an uploaded filename (drop dir + extension).

    ``"docs/meeting.md"`` -> ``"meeting"``; an empty / extension-only name falls
    back to ``"upload"`` so a source id is always non-empty.
    """
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return stem.strip() or "upload"


def _parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes] | None:
    """Extract ``(filename, file_bytes)`` from a ``multipart/form-data`` body.

    A small, dependency-free parser (``python-multipart`` is intentionally not a
    FRIDAY dependency): it locates the boundary from the ``Content-Type`` header,
    splits the body, and returns the first part that declares a ``filename`` in
    its ``Content-Disposition`` header. Returns ``None`` when no file part is
    present or the body is malformed, so the caller can answer ``422``.
    """
    marker = "boundary="
    idx = content_type.find(marker)
    if idx == -1:
        return None
    boundary = content_type[idx + len(marker) :].strip().strip('"')
    if not boundary:
        return None
    delimiter = b"--" + boundary.encode("latin-1")
    parts = body.split(delimiter)
    for part in parts:
        # Skip the preamble/epilogue and the closing ``--`` terminator.
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        trimmed = part.lstrip(b"\r\n")
        header_end = trimmed.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        raw_headers = trimmed[:header_end].decode("latin-1", errors="replace")
        if "filename=" not in raw_headers.lower():
            continue
        filename = _extract_filename(raw_headers)
        if filename is None:
            continue
        content = trimmed[header_end + 4 :]
        # The part content is terminated by a trailing CRLF before the next
        # boundary; strip exactly that closing newline.
        if content.endswith(b"\r\n"):
            content = content[:-2]
        return filename, content
    return None


def _extract_filename(raw_headers: str) -> str | None:
    """Pull the ``filename="..."`` value out of a part's header block."""
    for line in raw_headers.splitlines():
        lowered = line.lower()
        if "content-disposition" not in lowered or "filename=" not in lowered:
            continue
        key = "filename="
        start = lowered.find(key) + len(key)
        value = line[start:].strip()
        if value.startswith('"'):
            end = value.find('"', 1)
            if end != -1:
                return value[1:end]
        # Unquoted: take up to the next ``;`` or end of line.
        return value.split(";", 1)[0].strip().strip('"') or None
    return None


@router.post("/rag/ingest", response_model=None)
async def rag_ingest(request: Request) -> JSONResponse:
    """Ingest a note (JSON) or an uploaded file (multipart); 404 when disabled.

    JSON path: ``{source_id, text}`` -> chunk + index -> ``{source_id, chunks}``.
    Multipart path: the first uploaded file's bytes are decoded (``read_text``)
    and ingested under a source id derived from its filename. A body that is
    neither a valid JSON ingest request nor a multipart file upload is ``422``.
    """
    if not _rag_enabled(request):
        return _disabled()

    ingestor = _get_ingestor(request)
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        body = await request.body()
        parsed = _parse_multipart(body, content_type)
        if parsed is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "expected a multipart file upload"},
            )
        filename, data = parsed
        source_id = _source_id_from_filename(filename)
        text = ingestor.read_text(filename, data)
        result = await ingestor.ingest(source_id, text)
        return JSONResponse(status_code=200, content=result.model_dump())

    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        parsed_body = RagIngestRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    result = await ingestor.ingest(parsed_body.source_id, parsed_body.text)
    return JSONResponse(status_code=200, content=result.model_dump())


@router.get("/rag/sources", response_model=None)
async def rag_sources(request: Request) -> JSONResponse:
    """List the ingested source ids; 404 when RAG is disabled.

    Source ids are recovered from the long-term marker facts (each ``"ingested
    <source_id>"``). The list is deduplicated and order-stable (newest first, as
    the long-term store returns facts id-descending).
    """
    if not _rag_enabled(request):
        return _disabled()

    long_term = getattr(request.app.state, "long_term", None)
    sources: list[str] = []
    if long_term is not None:
        seen: set[str] = set()
        for fact in long_term.query_facts(_INGESTED_PREFIX, limit=_SOURCES_LIMIT):
            text = fact.text
            if not text.startswith(_INGESTED_PREFIX):
                continue
            source_id = text[len(_INGESTED_PREFIX) :].strip()
            if source_id and source_id not in seen:
                seen.add(source_id)
                sources.append(source_id)
    return JSONResponse(status_code=200, content={"sources": sources})


@router.delete("/rag/sources/{source_id}", response_model=None)
async def rag_forget(request: Request, source_id: str) -> JSONResponse:
    """Forget an ingested document from both stores; 404 when RAG is disabled.

    Returns ``{source_id, removed}`` with the number of rows dropped (0 when the
    source was never ingested — a no-op delete is still ``200``/idempotent).
    """
    if not _rag_enabled(request):
        return _disabled()
    ingestor = _get_ingestor(request)
    removed = ingestor.forget_source(source_id)
    return JSONResponse(
        status_code=200, content={"source_id": source_id, "removed": removed}
    )
