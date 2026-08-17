"""Unit tests for narrow wiring decisions made in :mod:`friday.app`.

These cover choices the app factory makes about which adapter to build, where
the decision is small, host-dependent, and easy to get quietly wrong — the kind
that otherwise only shows up as a feature silently doing nothing in production.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# The vault's OCR choice
# --------------------------------------------------------------------------- #
def test_vault_ocr_uses_tesseract_when_the_binary_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the photograph is the feature; use a real reader when we have one."""
    import importlib.util
    import shutil

    import friday.app as app_module

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    ocr = app_module._choose_vault_ocr()
    assert type(ocr).__name__ == "TesseractOCR"


def test_vault_ocr_falls_back_to_a_silent_fake_without_the_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tesseract on the host: file the capture unread rather than lose it."""
    import importlib.util
    import shutil

    import friday.app as app_module

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    ocr = app_module._choose_vault_ocr()
    assert type(ocr).__name__ == "FakeOCR"


def test_vault_ocr_falls_back_when_the_wrapper_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with the binary but no pytesseract must not pick the real reader."""
    import importlib.util
    import shutil

    import friday.app as app_module

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    ocr = app_module._choose_vault_ocr()
    assert type(ocr).__name__ == "FakeOCR"


# --------------------------------------------------------------------------- #
# Voice wiring must never take the app down
# --------------------------------------------------------------------------- #
def test_voice_wiring_survives_a_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag must not be able to crash boot.

    ``FasterWhisperSTT`` raises from its *constructor* when the voice extras are
    absent, so enabling voice on a host without them took the whole service down
    at startup — every other surface with it. Verified against a real deployment,
    which refused to boot with `_wire_voice` in the traceback.
    """
    from types import SimpleNamespace

    import friday.app as app_module
    from friday.errors import ProviderError

    def _explode(_settings: object) -> object:
        raise ProviderError("faster-whisper is not installed")

    monkeypatch.setattr(app_module, "_build_voice_stt", _explode)
    monkeypatch.setattr(app_module, "_build_voice_tts", _explode)

    app = SimpleNamespace(state=SimpleNamespace())
    settings = SimpleNamespace(enable_voice=True)

    app_module._wire_voice(app, settings)  # must not raise

    # Nothing half-built is left behind for a route to trip over.
    assert getattr(app.state, "voice_stt", None) is None
    assert getattr(app.state, "voice_tts", None) is None


def test_voice_wiring_keeps_the_half_that_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTS missing should not cost you transcription, or the reverse."""
    from types import SimpleNamespace

    import friday.app as app_module
    from friday.errors import ProviderError

    sentinel = object()
    monkeypatch.setattr(app_module, "_build_voice_stt", lambda _s: sentinel)
    monkeypatch.setattr(
        app_module, "_build_voice_tts",
        lambda _s: (_ for _ in ()).throw(ProviderError("no piper binary")),
    )

    app = SimpleNamespace(state=SimpleNamespace())
    app_module._wire_voice(app, SimpleNamespace(enable_voice=True))

    assert app.state.voice_stt is sentinel
    assert getattr(app.state, "voice_tts", None) is None
