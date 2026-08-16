"""OCR (image-to-text) boundary, deterministic fake, and lazy real adapter.

* :class:`OCRProvider` — the runtime-checkable ``read`` protocol.
* :class:`FakeOCR` — a deterministic provider returning scripted text, so tests
  run with zero heavy libraries and no models.
* :class:`TesseractOCR` — the real adapter that lazy-imports ``pytesseract`` +
  ``PIL.Image`` inside its method and raises a clear error (with a
  ``make install-perception`` hint) when the backend is absent.
* :class:`TrOCROCR` — the handwriting adapter over a fine-tuned TrOCR export,
  for the photographed-exercise-book case tesseract reads confidently wrong.

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

_TROCR_INSTALL_HINT = (
    "optimum / transformers / pillow are not installed. The handwriting OCR "
    "backend is optional and excluded from the uv lock; install it with "
    "`make install-perception` and export a model with "
    "`notebooks/friday_math_ocr.ipynb`."
)

#: Generation cap for a single line of handwriting. A photographed line that
#: decodes to more than this is a runaway, not a long line.
_TROCR_MAX_TOKENS = 64


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


class TrOCROCR:
    """Real :class:`OCRProvider` backed by a fine-tuned TrOCR export (lazy).

    Tesseract was built for printed text, and on handwriting it is not merely
    worse — it is wrong in a particular way, confidently returning ``S`` for
    ``5`` or dropping the stroke off a ``1``. That is the failure this adapter
    exists for: the vault's solver answers whatever the OCR hands it, so a
    misread digit becomes a confident answer to a question nobody asked.

    The model is the ONNX export from ``notebooks/friday_math_ocr.ipynb``.
    ``optimum``/``transformers`` are imported inside the method and the model is
    built once and reused, so importing this module costs nothing and a process
    that never selects this provider never loads it.
    """

    def __init__(self, model_dir: str) -> None:
        """Construct the adapter.

        Args:
            model_dir: Directory holding the exported ONNX model + processor
                (``models/ocr/trocr`` by convention).
        """
        self.model_dir = model_dir
        self._model: object | None = None
        self._processor: object | None = None

    def _ensure_loaded(self) -> tuple[object, object]:
        """Load the export once; subsequent reads reuse it."""
        if self._model is None or self._processor is None:
            try:
                # Optional perception backend, excluded from the uv lock.
                from optimum.onnxruntime import (  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
                    ORTModelForVision2Seq,
                )
                from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
                    TrOCRProcessor,
                )
            except ImportError as exc:
                raise ProviderError(_TROCR_INSTALL_HINT) from exc
            self._processor = TrOCRProcessor.from_pretrained(self.model_dir)
            self._model = ORTModelForVision2Seq.from_pretrained(self.model_dir)
        return self._model, self._processor

    async def read(self, image: bytes) -> str:
        """Decode ``image`` and run the fine-tuned model, returning its text."""
        from io import BytesIO

        try:
            import PIL.Image  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_TROCR_INSTALL_HINT) from exc

        model, processor = self._ensure_loaded()
        with PIL.Image.open(BytesIO(image)) as img:
            pixels = processor(img.convert("RGB"), return_tensors="pt").pixel_values  # type: ignore[operator]
            generated = model.generate(pixels, max_length=_TROCR_MAX_TOKENS)  # type: ignore[attr-defined]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)  # type: ignore[attr-defined]
        return str(decoded[0]).strip() if decoded else ""
