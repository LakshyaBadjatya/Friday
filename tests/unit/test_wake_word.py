"""Unit tests for the wake-word detection boundary, fake, and lazy adapter.

Pins the A2 contract from the Phase 3 plan: a positive fixture frame is detected
above threshold, a negative frame below it (the precision boundary is asserted),
and the real ``openwakeword`` adapter raises a helpful error when its backend is
missing. No real audio, no network.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from friday.errors import ProviderError
from friday.voice.fixtures import make_silence_frame, make_wake_frame
from friday.voice.wake_word import (
    DEFAULT_WAKE_THRESHOLD,
    FakeWakeWord,
    OpenWakeWordDetector,
    WakeResult,
    WakeWordDetector,
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_wake_result_constructs() -> None:
    r = WakeResult(detected=True, score=0.9)
    assert r.detected is True
    assert r.score == 0.9


def test_wake_result_score_bounds_enforced() -> None:
    with pytest.raises(ValueError):
        WakeResult(detected=False, score=1.5)
    with pytest.raises(ValueError):
        WakeResult(detected=False, score=-0.1)


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fake_is_detector() -> None:
    assert isinstance(FakeWakeWord(), WakeWordDetector)


# --------------------------------------------------------------------------- #
# Acceptance: positive above threshold, negative below
# --------------------------------------------------------------------------- #
def test_positive_fixture_detected_above_threshold() -> None:
    detector = FakeWakeWord()
    result = detector.detect(make_wake_frame())
    assert result.detected is True
    # Precision boundary: a positive must land at or above the threshold.
    assert result.score >= detector.threshold


def test_negative_fixture_below_threshold() -> None:
    detector = FakeWakeWord()
    result = detector.detect(make_silence_frame())
    assert result.detected is False
    # Precision boundary: a negative must land strictly below the threshold.
    assert result.score < detector.threshold


def test_plain_tone_frame_is_negative() -> None:
    # A non-marked, non-silent frame must still be a negative (no false fire).
    from friday.voice.fixtures import make_tone_frame

    detector = FakeWakeWord()
    result = detector.detect(make_tone_frame())
    assert result.detected is False
    assert result.score < detector.threshold


def test_threshold_boundary_with_custom_threshold() -> None:
    detector = FakeWakeWord(threshold=0.8)
    assert detector.threshold == 0.8
    assert detector.detect(make_wake_frame()).score >= 0.8
    assert detector.detect(make_silence_frame()).score < 0.8


def test_default_threshold_used_without_config_field() -> None:
    # With no FRIDAY_WAKE_WORD_THRESHOLD set, the fallback default applies.
    detector = FakeWakeWord()
    assert detector.threshold == DEFAULT_WAKE_THRESHOLD


# --------------------------------------------------------------------------- #
# Real adapter: lazy import, helpful error when backend missing
# --------------------------------------------------------------------------- #
def test_openwakeword_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openwakeword" or name.startswith("openwakeword."):
            raise ImportError("No module named 'openwakeword'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError) as exc:
        OpenWakeWordDetector()
    assert "install-voice" in str(exc.value)


def test_module_import_does_not_require_openwakeword() -> None:
    # Importing the module must not pull in the heavy backend.
    import importlib
    import sys

    assert "openwakeword" not in sys.modules
    importlib.import_module("friday.voice.wake_word")
    assert "openwakeword" not in sys.modules
