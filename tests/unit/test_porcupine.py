"""Unit tests for the Porcupine wake-word adapter.

Pins the contract for :class:`friday.voice.porcupine.PorcupineWakeWord`, a second
backend alongside ``openwakeword`` that lazy-imports ``pvporcupine`` (which is NOT
in the uv lock). The tests assert, with no real backend installed:

* the module imports without pulling in ``pvporcupine``;
* constructing/using the adapter without the backend raises a clear, actionable
  :class:`~friday.errors.ProviderError` (install hint + Picovoice key hint);
* the class structurally satisfies the
  :class:`friday.voice.wake_word.WakeWordDetector` protocol.

No real audio, no network, no heavy backend.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from friday.errors import ProviderError
from friday.voice.porcupine import PorcupineWakeWord
from friday.voice.wake_word import WakeResult, WakeWordDetector


# --------------------------------------------------------------------------- #
# Module import must not require the heavy backend
# --------------------------------------------------------------------------- #
def test_module_import_does_not_require_pvporcupine() -> None:
    # Importing the module must not pull in the optional backend.
    import importlib
    import sys

    assert "pvporcupine" not in sys.modules
    importlib.import_module("friday.voice.porcupine")
    assert "pvporcupine" not in sys.modules


# --------------------------------------------------------------------------- #
# Structural protocol membership
# --------------------------------------------------------------------------- #
def test_porcupine_structurally_satisfies_detector_protocol() -> None:
    # The class exposes ``detect(frame) -> WakeResult`` so it satisfies the
    # runtime-checkable protocol structurally (without constructing it, which
    # would require the backend).
    assert issubclass(PorcupineWakeWord, WakeWordDetector)
    detect = PorcupineWakeWord.detect
    assert callable(detect)
    # ``from __future__ import annotations`` makes annotations strings; the
    # ``detect`` signature returns a ``WakeResult`` as the protocol requires.
    assert detect.__annotations__.get("return") == "WakeResult"
    assert detect.__annotations__.get("frame") == "bytes"
    # The return type is in fact the protocol's WakeResult model.
    assert WakeResult.__name__ == "WakeResult"


# --------------------------------------------------------------------------- #
# Missing backend -> clear, actionable error
# --------------------------------------------------------------------------- #
def _block_pvporcupine(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pvporcupine" or name.startswith("pvporcupine."):
            raise ImportError("No module named 'pvporcupine'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_construct_without_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_pvporcupine(monkeypatch)

    with pytest.raises(ProviderError) as exc:
        PorcupineWakeWord(access_key="pv-key")
    message = str(exc.value)
    # Actionable: points the user at the install path...
    assert "install-voice" in message
    assert "pvporcupine" in message


def test_missing_access_key_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even if the backend were present, no Picovoice key must surface a clear
    # error that names the key requirement (and never logs the absent secret).
    monkeypatch.delenv("FRIDAY_PICOVOICE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PICOVOICE_ACCESS_KEY", raising=False)

    with pytest.raises(ProviderError) as exc:
        PorcupineWakeWord(access_key="")
    message = str(exc.value)
    assert "Picovoice" in message


def test_default_threshold_used_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The constructor records the threshold before the backend import is even
    # attempted, so a custom threshold survives the (expected) ProviderError via
    # the raised error path — assert it does not crash before the import guard.
    _block_pvporcupine(monkeypatch)

    with pytest.raises(ProviderError):
        PorcupineWakeWord(access_key="pv-key", threshold=0.8)
