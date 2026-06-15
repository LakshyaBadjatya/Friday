"""Screen-capture boundary, fixture fake, lazy real adapter, and the service.

* :class:`ScreenCapture` — the runtime-checkable ``capture`` protocol.
* :class:`FakeScreen` — returns deterministic fixture bytes, so tests run with
  zero heavy libraries and no display.
* :class:`MssScreen` — the real adapter that lazy-imports ``mss`` + ``PIL`` inside
  its method and raises a clear error (with a ``make install-perception`` hint)
  when the backend is absent.
* :class:`PerceptionService` — composes a screen capture with OCR + vision into
  ``describe_screen()`` (capture -> ocr + detect).

No heavy perception library is imported at module top level, so importing this
module never requires ``mss``/``pillow`` and ``uv sync`` stays unaffected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from friday.errors import ProviderError
from friday.perception.clipboard import ClipboardProvider
from friday.perception.ocr import OCRProvider
from friday.perception.vision import Detection, VisionProvider

_INSTALL_HINT = (
    "mss / pillow are not installed. Perception extras are optional and excluded "
    "from the uv lock; install them with `make install-perception`."
)

# Deterministic fixture bytes the fake returns; an arbitrary, recognizable
# marker so a test can assert the capture seam fed exactly this into OCR/vision.
FIXTURE_SCREEN_BYTES = b"FRIDAY_FAKE_SCREEN_CAPTURE"


@runtime_checkable
class ScreenCapture(Protocol):
    """Contract capturing the current screen as encoded image bytes."""

    def capture(self) -> bytes:
        """Capture the screen and return encoded image bytes (e.g. PNG)."""
        ...


class FakeScreen:
    """A deterministic :class:`ScreenCapture` for tests.

    Returns fixed :data:`FIXTURE_SCREEN_BYTES` (or a caller-supplied payload) so
    the capture seam is exercised with no display or screenshot backend.
    """

    def __init__(self, payload: bytes = FIXTURE_SCREEN_BYTES) -> None:
        """Create the fake screen.

        Args:
            payload: The bytes returned by :meth:`capture`.
        """
        self.payload = payload

    def capture(self) -> bytes:
        return self.payload


class MssScreen:
    """Real :class:`ScreenCapture` backed by ``mss`` (lazy).

    The heavy ``mss`` + ``PIL`` imports happen inside :meth:`capture`, so
    importing this module never requires the backend. When the backend is
    missing, a :class:`friday.errors.ProviderError` is raised with a
    ``make install-perception`` hint.
    """

    def __init__(self, monitor: int = 1) -> None:
        """Construct the screen-capture adapter.

        Args:
            monitor: The ``mss`` monitor index to grab (1 is the primary).
        """
        self.monitor = monitor

    def capture(self) -> bytes:
        """Grab the configured monitor and return PNG-encoded bytes."""
        from io import BytesIO

        try:
            # Optional perception backend: excluded from the uv lock, so mypy has
            # no stub for it; lazily imported here and guarded by the ImportError.
            # Imported as plain modules (not ``from PIL import Image``) so the
            # whole-statement ``# type: ignore`` stays on one line ruff won't wrap.
            import mss  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
            import PIL.Image  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[self.monitor])
            img = PIL.Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


class ScreenDescription(BaseModel):
    """The combined result of describing a captured screen.

    Attributes:
        ocr_text: The text OCR read from the captured image.
        detections: The objects vision detected in the captured image.
    """

    ocr_text: str
    detections: list[Detection]


class PerceptionService:
    """Compose screen capture with OCR + vision into one description pass.

    :meth:`describe_screen` captures the screen once and feeds that single image
    into both the OCR and vision providers, returning a :class:`ScreenDescription`
    bundling the read text and the detected objects. All four collaborators are
    injected, so the offline default is composed entirely from fakes and the
    service needs no heavy library.
    """

    def __init__(
        self,
        vision: VisionProvider,
        ocr: OCRProvider,
        clipboard: ClipboardProvider,
        screen: ScreenCapture,
    ) -> None:
        """Assemble the service from its four collaborators.

        Args:
            vision: The object-detection provider.
            ocr: The image-to-text provider.
            clipboard: The clipboard provider (surfaced for the routes; not used
                by :meth:`describe_screen`).
            screen: The screen-capture provider.
        """
        self.vision = vision
        self.ocr = ocr
        self.clipboard = clipboard
        self.screen = screen

    async def describe_screen(self) -> ScreenDescription:
        """Capture the screen and describe it via OCR + object detection.

        Returns:
            A :class:`ScreenDescription` with the OCR text and the detections,
            both derived from the *same* captured image.
        """
        image = self.screen.capture()
        ocr_text = await self.ocr.read(image)
        detections = await self.vision.detect(image)
        return ScreenDescription(ocr_text=ocr_text, detections=detections)
