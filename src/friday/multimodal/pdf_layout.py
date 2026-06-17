# © Lakshya Badjatya — Author
"""Layout-aware PDF extraction seam — bytes in, pages of text blocks out.

A runtime-checkable :class:`PdfLayoutExtractor` protocol with a dependency-free
:class:`FakePdfLayoutExtractor` (decodes the bytes as text and splits on blank
lines into blocks — deterministic, perfect for offline tests) and a lazily
imported :class:`PyMuPdfLayoutExtractor` real backend. :func:`build_pdf_extractor`
returns ``None`` unless ``enable_pdf_layout`` is on and degrades to the fake when
PyMuPDF is absent.

The structured :class:`PdfDocument` (pages → text blocks) is what personal-RAG
ingestion chunks instead of one flat string, preserving layout boundaries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from friday.config import Settings

logger = logging.getLogger("friday.multimodal.pdf_layout")


class PdfPage(BaseModel):
    """One page: its 1-based number and the ordered text blocks on it."""

    number: int
    blocks: list[str] = Field(default_factory=list)


class PdfDocument(BaseModel):
    """A parsed document: its pages, each a list of text blocks."""

    pages: list[PdfPage] = Field(default_factory=list)


@runtime_checkable
class PdfLayoutExtractor(Protocol):
    """Anything that turns PDF bytes into a structured :class:`PdfDocument`."""

    def extract(self, pdf_bytes: bytes) -> PdfDocument:
        """Extract pages + text blocks from ``pdf_bytes``."""
        ...


class FakePdfLayoutExtractor:
    """Deterministic extractor: decode bytes as text, split on blank lines.

    Treats the input as a single page whose blocks are the blank-line-delimited
    paragraphs — so the layout contract is exercised with no PDF backend (handy
    for tests that pass plain text bytes).
    """

    def extract(self, pdf_bytes: bytes) -> PdfDocument:
        """Return a one-page document of blank-line-delimited blocks."""
        text = pdf_bytes.decode("utf-8", errors="replace")
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        return PdfDocument(pages=[PdfPage(number=1, blocks=blocks)])


class PyMuPdfLayoutExtractor:
    """Real layout extractor over PyMuPDF (``fitz``), lazy-imported in :meth:`extract`."""

    def extract(self, pdf_bytes: bytes) -> PdfDocument:
        """Open the PDF from bytes and read each page's text blocks."""
        fitz = _load_pymupdf()
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[PdfPage] = []
        for index, page in enumerate(document):
            raw_blocks = page.get_text("blocks")
            blocks = [b[4].strip() for b in raw_blocks if b[4].strip()]
            pages.append(PdfPage(number=index + 1, blocks=blocks))
        return PdfDocument(pages=pages)


def _load_pymupdf() -> Any:
    """Lazy-import PyMuPDF; raises ImportError when not installed."""
    import fitz  # type: ignore[import-not-found]  # noqa: PLC0415

    return fitz


def _pymupdf_available() -> bool:
    """Whether PyMuPDF (``fitz``) can be imported."""
    try:
        import fitz  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def build_pdf_extractor(settings: Settings) -> PdfLayoutExtractor | None:
    """Return an extractor when ``enable_pdf_layout`` is on, else ``None``.

    Real PyMuPDF backend when installed, otherwise the deterministic text fake
    (logged) — so the flag is safe without the native PDF library.
    """
    if not settings.enable_pdf_layout:
        return None
    if _pymupdf_available():
        return PyMuPdfLayoutExtractor()
    logger.warning(
        "enable_pdf_layout is set but PyMuPDF is not installed; using the "
        "text-only FakePdfLayoutExtractor (install pdf extras for real layout)"
    )
    return FakePdfLayoutExtractor()
