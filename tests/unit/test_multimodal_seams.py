# © Lakshya Badjatya — Author
"""Unit tests for the image-gen + PDF-layout seams (flag-gated, lazy backends)."""

from __future__ import annotations

import base64

from friday.config import Settings
from friday.multimodal.imagegen import (
    FakeImageGenerator,
    GeneratedImage,
    ImageGenerator,
    build_image_generator,
)
from friday.multimodal.pdf_layout import (
    FakePdfLayoutExtractor,
    PdfDocument,
    PdfLayoutExtractor,
    build_pdf_extractor,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", **overrides)  # type: ignore[arg-type]


# --- image generation ------------------------------------------------------- #
def test_fake_image_generator_returns_svg_with_prompt() -> None:
    img = FakeImageGenerator().generate("a glowing arc reactor")
    assert isinstance(img, GeneratedImage)
    assert img.media_type == "image/svg+xml"
    decoded = base64.b64decode(img.data_base64).decode("utf-8")
    assert "<svg" in decoded
    assert "a glowing arc reactor" in decoded


def test_fake_image_generator_satisfies_protocol() -> None:
    assert isinstance(FakeImageGenerator(), ImageGenerator)


def test_build_image_generator_gating_and_fallback() -> None:
    assert build_image_generator(_settings()) is None
    # diffusers absent here -> enabled build degrades to the SVG fake.
    built = build_image_generator(_settings(enable_imagegen=True))
    assert isinstance(built, FakeImageGenerator)


# --- PDF layout ------------------------------------------------------------- #
def test_fake_pdf_extractor_splits_blocks_on_blank_lines() -> None:
    raw = b"Title block\n\nFirst paragraph.\n\nSecond paragraph."
    doc = FakePdfLayoutExtractor().extract(raw)
    assert isinstance(doc, PdfDocument)
    assert len(doc.pages) == 1
    assert doc.pages[0].number == 1
    assert doc.pages[0].blocks == ["Title block", "First paragraph.", "Second paragraph."]


def test_fake_pdf_extractor_satisfies_protocol() -> None:
    assert isinstance(FakePdfLayoutExtractor(), PdfLayoutExtractor)


def test_build_pdf_extractor_gating_and_fallback() -> None:
    assert build_pdf_extractor(_settings()) is None
    # PyMuPDF absent here -> enabled build degrades to the text fake.
    built = build_pdf_extractor(_settings(enable_pdf_layout=True))
    assert isinstance(built, FakePdfLayoutExtractor)
