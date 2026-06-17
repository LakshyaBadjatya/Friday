# © Lakshya Badjatya — Author
"""``/imagegen`` + ``/pdf/layout`` — flagged multimodal seams (Wave C; default off).

Two independent, self-guarding endpoints:

* ``POST /imagegen`` — render a text prompt to an image (base64), gated by
  ``FRIDAY_ENABLE_IMAGEGEN``. The offline fake returns a deterministic SVG
  placeholder; the real ``diffusers`` backend (when installed) returns a PNG.
* ``POST /pdf/layout`` — extract a base64 PDF into pages of text blocks, gated by
  ``FRIDAY_ENABLE_PDF_LAYOUT``. The offline fake decodes the bytes as text; the
  real PyMuPDF backend (when installed) reads true layout blocks.

Each reads its flag lazily from :func:`~friday.config.get_settings` and is ``404``
when off, so the router works mounted on a bare ``FastAPI()`` and the feature
"does not exist" until enabled. The seams are built per-request from settings
(the offline fakes are trivially cheap); a real backend would be cached on
``app.state`` in production.
"""

from __future__ import annotations

import base64
import binascii

import anyio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.config import get_settings
from friday.multimodal.imagegen import build_image_generator
from friday.multimodal.pdf_layout import build_pdf_extractor

router = APIRouter()


class ImageGenRequest(BaseModel):
    """The ``POST /imagegen`` body: the text prompt to render."""

    prompt: str = Field(min_length=1, max_length=2_000)


class PdfLayoutRequest(BaseModel):
    """The ``POST /pdf/layout`` body: base64-encoded PDF bytes."""

    pdf_base64: str = Field(min_length=1)


@router.post("/imagegen", response_model=None)
async def generate_image(body: ImageGenRequest) -> JSONResponse:
    """Render the prompt to an image; ``404`` when image generation is disabled."""
    generator = build_image_generator(get_settings())
    if generator is None:
        return JSONResponse(status_code=404, content={"detail": "imagegen disabled"})
    # generate() lazily loads a (multi-GB) diffusion pipeline and runs CPU/GPU
    # inference — offload it so a single request cannot stall the event loop.
    image = await anyio.to_thread.run_sync(generator.generate, body.prompt)
    return JSONResponse(status_code=200, content=image.model_dump())


@router.post("/pdf/layout", response_model=None)
async def extract_pdf_layout(body: PdfLayoutRequest) -> JSONResponse:
    """Extract a base64 PDF into pages of text blocks; ``404`` when disabled."""
    extractor = build_pdf_extractor(get_settings())
    if extractor is None:
        return JSONResponse(status_code=404, content={"detail": "pdf layout disabled"})
    try:
        pdf_bytes = base64.b64decode(body.pdf_base64, validate=True)
    except (ValueError, binascii.Error):
        return JSONResponse(
            status_code=400,
            content={"detail": "pdf_base64 is not valid base64", "type": "BadInput"},
        )
    # PDF parsing is synchronous, CPU-bound work over arbitrarily large input;
    # offload it so it does not block the event loop / other in-flight requests.
    document = await anyio.to_thread.run_sync(extractor.extract, pdf_bytes)
    return JSONResponse(status_code=200, content=document.model_dump())
