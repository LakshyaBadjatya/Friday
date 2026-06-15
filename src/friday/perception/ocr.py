"""OCR (image-to-text) boundary, deterministic fake, and lazy real adapter.

* :class:`OCRProvider` — the runtime-checkable ``read`` protocol.
* :class:`FakeOCR` — a deterministic provider returning scripted text, so tests
  run with zero heavy libraries and no models.
* :class:`TesseractOCR` — the real adapter that lazy-imports ``pytesseract`` +
  ``PIL.Image`` inside its method and raises a clear error (with a
  ``make install-perception`` hint) when the backend is absent.

No heavy perception library is imported at module top level, so importing this
module never requires ``pytesseract``/``pillow`` and ``uv sync`` stays unaffected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from friday.errors import ProviderError

_INSTALL_HINT = (
    "pytesseract / pillow are not installed. Perception extras are optional and "
    "excluded from the uv lock; install them with `make install-perception` "
    "(the tesseract binary must also be present on the host)."
)


@runtime_checkable
class OCRProvider(Protocol):
    """Contract reading text out of an ``image`` (encoded image bytes)."""

    async def read(self, image: bytes) -> str:
        """Read the text content of ``image``.

        Args:
            image: Encoded image bytes (e.g. PNG/JPEG).

        Returns:
            The extracted text (possibly empty).
        """
        ...


class FakeOCR:
    """A deterministic :class:`OCRProvider` for tests.

    Returns its scripted ``text`` for every call, so the offline path exercises a
    deterministic read with no models or binaries.
    """

    def __init__(self, text: str = "hello world") -> None:
        """Create the fake provider.

        Args:
            text: The text returned by :meth:`read`.
        """
        self.text = text

    async def read(self, image: bytes) -> str:
        return self.text


class TesseractOCR:
    """Real :class:`OCRProvider` backed by ``pytesseract`` (lazy).

    The heavy ``pytesseract`` + ``PIL.Image`` imports happen inside :meth:`read`,
    so importing this module never requires the backend. When the backend is
    missing, a :class:`friday.errors.ProviderError` is raised with a
    ``make install-perception`` hint.
    """

    def __init__(self, lang: str = "eng") -> None:
        """Construct the OCR adapter.

        Args:
            lang: The tesseract language code passed at read time.
        """
        self.lang = lang

    async def read(self, image: bytes) -> str:
        """Decode ``image`` and run tesseract, returning the extracted text."""
        from io import BytesIO

        try:
            # Optional perception backend: excluded from the uv lock, so mypy has
            # no stub for it; lazily imported here and guarded by the ImportError.
            # Imported as plain modules (not ``from PIL import Image``) so the
            # whole-statement ``# type: ignore`` stays on one line ruff won't wrap.
            import PIL.Image  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
            import pytesseract  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        with PIL.Image.open(BytesIO(image)) as img:
            text = pytesseract.image_to_string(img, lang=self.lang)
        return str(text).strip()
