# © Lakshya Badjatya — Author
"""Unit tests for the wake-word engine seam (Fake engine + threshold)."""

from __future__ import annotations

from friday.voice.wake_engine import (
    DEFAULT_WAKE_THRESHOLD,
    FakeWakeWordEngine,
    WakeWordEngine,
    detected,
)


def test_fake_engine_pops_scripted_scores_then_zero() -> None:
    engine = FakeWakeWordEngine([0.1, 0.9])
    assert engine.score(b"") == 0.1
    assert engine.score(b"") == 0.9
    assert engine.score(b"") == 0.0  # exhausted -> silence


def test_fake_engine_satisfies_protocol() -> None:
    assert isinstance(FakeWakeWordEngine(), WakeWordEngine)


def test_detected_threshold_is_boundary_inclusive() -> None:
    assert detected(0.5) is True  # == default threshold
    assert detected(0.49) is False
    assert detected(0.9, threshold=0.95) is False
    assert detected(0.96, threshold=0.95) is True


def test_default_threshold_value() -> None:
    assert DEFAULT_WAKE_THRESHOLD == 0.5
