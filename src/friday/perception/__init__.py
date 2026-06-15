"""Perception subsystem: vision, OCR, clipboard, and screen capture.

This package owns the typed boundaries for FRIDAY's perception inputs — object
detection, optical character recognition, the system clipboard, and screen
capture — plus the :class:`~friday.perception.screen.PerceptionService` that
composes them into a single ``describe_screen()`` pass (capture -> ocr + detect).

Perception is **privacy-heavy** (it can read your screen and clipboard) and is
therefore **off by default**, gated behind ``FRIDAY_ENABLE_PERCEPTION``. All real
adapters lazy-import their heavy backend (``opencv-python``/``ultralytics``,
``pytesseract``/``pillow``, ``pyperclip``, ``mss``/``pillow``) *inside*
methods/``__init__`` and raise a clear error with a ``make install-perception``
hint when the backend is missing, so importing this package never requires any
heavy perception library and ``uv sync`` stays unaffected.

No LLM SDK is imported anywhere in this package (architecture guard).
"""

from __future__ import annotations

from friday.perception.clipboard import (
    ClipboardProvider,
    FakeClipboard,
    SystemClipboard,
)
from friday.perception.ocr import FakeOCR, OCRProvider, TesseractOCR
from friday.perception.screen import (
    FakeScreen,
    MssScreen,
    PerceptionService,
    ScreenCapture,
    ScreenDescription,
)
from friday.perception.vision import (
    Detection,
    FakeVision,
    VisionProvider,
    YoloVision,
)

__all__ = [
    "ClipboardProvider",
    "Detection",
    "FakeClipboard",
    "FakeOCR",
    "FakeScreen",
    "FakeVision",
    "MssScreen",
    "OCRProvider",
    "PerceptionService",
    "ScreenCapture",
    "ScreenDescription",
    "SystemClipboard",
    "TesseractOCR",
    "VisionProvider",
    "YoloVision",
]
