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
