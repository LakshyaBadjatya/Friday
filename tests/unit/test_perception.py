"""Unit tests for the perception boundaries, fakes, and lazy real adapters.

Pins the perception contract: importing every perception module requires NO
heavy library; each real adapter raises a clear ``make install-perception`` error
when its backend is absent (the import is monkeypatched to fail); the fakes
behave deterministically; and ``PerceptionService.describe_screen`` combines OCR
text + detections derived from the *same* captured image. No real screen, no
models, no network.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from friday.errors import ProviderError
from friday.perception.clipboard import (
    ClipboardProvider,
    FakeClipboard,
    SystemClipboard,
)
from friday.perception.ocr import FakeOCR, OCRProvider, TesseractOCR, TrOCROCR
from friday.perception.screen import (
    FIXTURE_SCREEN_BYTES,
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


def _fail_import_of(*modules: str) -> Any:
    """Build a fake ``__import__`` that raises ImportError for ``modules``."""
    real_import = builtins.__import__
    blocked = tuple(modules)

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in blocked or any(name.startswith(f"{m}.") for m in blocked):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    return fake_import


# --------------------------------------------------------------------------- #
# Importing the modules requires NO heavy library
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module",
    [
        "friday.perception",
        "friday.perception.vision",
        "friday.perception.ocr",
        "friday.perception.clipboard",
        "friday.perception.screen",
    ],
)
def test_module_import_requires_no_heavy_lib(module: str) -> None:
    """Importing the module must not drag a heavy library in with it.

    The heavy libraries and the module under test are both evicted from
    ``sys.modules`` first, so this asks what it means to ask. Checking the live
    ``sys.modules`` alone only held while no earlier test in the run had
    imported one of them — it passed by luck of alphabetical ordering, and any
    unrelated test that touched Pillow turned it red.
    """
    heavy_libs = ("cv2", "ultralytics", "pytesseract", "PIL", "mss", "pyperclip")

    def _owned_by(names: tuple[str, ...]) -> list[str]:
        return [
            loaded
            for loaded in list(sys.modules)
            if loaded in names or any(loaded.startswith(f"{n}.") for n in names)
        ]

    evicted = {name: sys.modules.pop(name) for name in _owned_by(heavy_libs)}
    for name in _owned_by((module,)):
        del sys.modules[name]
    try:
        importlib.import_module(module)
        for heavy in heavy_libs:
            assert heavy not in sys.modules, f"{module} pulled in heavy lib {heavy}"
    finally:
        sys.modules.update(evicted)


# --------------------------------------------------------------------------- #
# Detection model
# --------------------------------------------------------------------------- #
def test_detection_constructs() -> None:
    d = Detection(label="cat", confidence=0.8, bbox=(1.0, 2.0, 3.0, 4.0))
    assert d.label == "cat"
    assert d.confidence == 0.8
    assert d.bbox == (1.0, 2.0, 3.0, 4.0)


def test_detection_confidence_bounds_enforced() -> None:
    with pytest.raises(ValueError):
        Detection(label="x", confidence=1.5, bbox=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Detection(label="x", confidence=-0.1, bbox=(0.0, 0.0, 1.0, 1.0))


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeVision(), VisionProvider)
    assert isinstance(FakeOCR(), OCRProvider)
    assert isinstance(FakeClipboard(), ClipboardProvider)
    assert isinstance(FakeScreen(), ScreenCapture)


# --------------------------------------------------------------------------- #
# FakeVision
# --------------------------------------------------------------------------- #
async def test_fake_vision_default_script() -> None:
    detections = await FakeVision().detect(b"image-bytes")
    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].confidence == 0.9


async def test_fake_vision_scripted_detections() -> None:
    scripted = [
        Detection(label="dog", confidence=0.7, bbox=(0.0, 0.0, 10.0, 10.0)),
        Detection(label="ball", confidence=0.5, bbox=(5.0, 5.0, 8.0, 8.0)),
    ]
    vision = FakeVision(scripted)
    out = await vision.detect(b"anything")
    assert [d.label for d in out] == ["dog", "ball"]
    # A returned copy: mutating it does not corrupt the script.
    out.clear()
    assert [d.label for d in await vision.detect(b"again")] == ["dog", "ball"]


# --------------------------------------------------------------------------- #
# FakeOCR
# --------------------------------------------------------------------------- #
async def test_fake_ocr_default_and_scripted() -> None:
    assert await FakeOCR().read(b"img") == "hello world"
    assert await FakeOCR("scripted text").read(b"img") == "scripted text"


# --------------------------------------------------------------------------- #
# FakeClipboard
# --------------------------------------------------------------------------- #
def test_fake_clipboard_round_trip() -> None:
    clip = FakeClipboard()
    assert clip.read() == ""
    clip.write("copied!")
    assert clip.read() == "copied!"


def test_fake_clipboard_initial_value() -> None:
    assert FakeClipboard("seed").read() == "seed"


# --------------------------------------------------------------------------- #
# FakeScreen
# --------------------------------------------------------------------------- #
def test_fake_screen_returns_fixture_bytes() -> None:
    assert FakeScreen().capture() == FIXTURE_SCREEN_BYTES
    assert FakeScreen(b"custom").capture() == b"custom"


# --------------------------------------------------------------------------- #
# PerceptionService.describe_screen combines ocr + detections from fakes
# --------------------------------------------------------------------------- #
async def test_describe_screen_combines_ocr_and_detections() -> None:
    scripted = [Detection(label="window", confidence=0.6, bbox=(0.0, 0.0, 2.0, 2.0))]
    service = PerceptionService(
        vision=FakeVision(scripted),
        ocr=FakeOCR("on-screen text"),
        clipboard=FakeClipboard(),
        screen=FakeScreen(),
    )
    description = await service.describe_screen()
    assert isinstance(description, ScreenDescription)
    assert description.ocr_text == "on-screen text"
    assert [d.label for d in description.detections] == ["window"]


async def test_describe_screen_feeds_captured_image_to_both() -> None:
    """The OCR + vision providers receive the bytes the screen captured."""
    seen: dict[str, bytes] = {}

    class _RecordingOCR:
        async def read(self, image: bytes) -> str:
            seen["ocr"] = image
            return "x"

    class _RecordingVision:
        async def detect(self, image: bytes) -> list[Detection]:
            seen["vision"] = image
            return []

    service = PerceptionService(
        vision=_RecordingVision(),
        ocr=_RecordingOCR(),
        clipboard=FakeClipboard(),
        screen=FakeScreen(b"PAYLOAD"),
    )
    await service.describe_screen()
    assert seen["ocr"] == b"PAYLOAD"
    assert seen["vision"] == b"PAYLOAD"


# --------------------------------------------------------------------------- #
# Real adapters: lazy import, helpful error when backend missing
# --------------------------------------------------------------------------- #
def test_yolo_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("cv2", "ultralytics"))
    with pytest.raises(ProviderError) as exc:
        YoloVision()
    assert "install-perception" in str(exc.value)


async def test_tesseract_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lazy import is inside read(); construction itself stays import-light.
    ocr = TesseractOCR()
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("pytesseract", "PIL"))
    with pytest.raises(ProviderError) as exc:
        await ocr.read(b"image")
    assert "install-perception" in str(exc.value)


def test_system_clipboard_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = SystemClipboard()
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("pyperclip"))
    with pytest.raises(ProviderError) as read_exc:
        clip.read()
    assert "install-perception" in str(read_exc.value)
    with pytest.raises(ProviderError) as write_exc:
        clip.write("x")
    assert "install-perception" in str(write_exc.value)


def test_mss_screen_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = MssScreen()
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("mss", "PIL"))
    with pytest.raises(ProviderError) as exc:
        screen.capture()
    assert "install-perception" in str(exc.value)


def test_trocr_satisfies_the_ocr_protocol() -> None:
    """The point of the adapter: the pipeline swaps it in with no other change."""
    assert isinstance(TrOCROCR("models/ocr/trocr"), OCRProvider)


async def test_trocr_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr = TrOCROCR("models/ocr/trocr")
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("optimum", "transformers", "PIL"))
    with pytest.raises(ProviderError) as exc:
        await ocr.read(b"image")
    assert "install-perception" in str(exc.value)


async def test_trocr_returns_the_decoded_line_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoded line comes back trimmed, not padded with the model's spacing."""

    class _Processor:
        def __call__(self, image: Any, return_tensors: str = "pt") -> Any:
            return SimpleNamespace(pixel_values="pixels")

        def batch_decode(self, generated: Any, skip_special_tokens: bool = True) -> list[str]:
            return ["  2x + 1  "]

    class _Model:
        def generate(self, pixels: Any, max_length: int = 0) -> str:
            return "tokens"

    ocr = TrOCROCR("models/ocr/trocr")

    def _load() -> tuple[object, object]:
        return _Model(), _Processor()

    monkeypatch.setattr(ocr, "_ensure_loaded", _load)

    class _Image:
        def convert(self, mode: str) -> str:
            return "rgb"

        def __enter__(self) -> _Image:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(
        "PIL.Image.open", lambda _buffer: _Image(), raising=False
    )
    assert await ocr.read(b"image") == "2x + 1"


async def test_trocr_caches_the_loaded_model() -> None:
    """_ensure_loaded builds once and hands the same objects back."""
    ocr = TrOCROCR("models/ocr/trocr")
    ocr._model = object()
    ocr._processor = object()
    first = ocr._ensure_loaded()
    second = ocr._ensure_loaded()
    assert first[0] is second[0]
    assert first[1] is second[1]
