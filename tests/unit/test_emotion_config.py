"""Emotion feature flags on Settings (Phase 1)."""

from __future__ import annotations

import pytest

from friday.config import Settings


def test_emotion_flags_default_off() -> None:
    s = Settings(_env_file=None)
    assert s.enable_emotion is False
    assert s.emotion_provider == "lite"
    assert s.emotion_model == "models/emotion/emotion_head.onnx"
    assert s.emotion_adapt is False


def test_emotion_flags_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_EMOTION", "true")
    monkeypatch.setenv("FRIDAY_EMOTION_PROVIDER", "fake")
    s = Settings(_env_file=None)
    assert s.enable_emotion is True
    assert s.emotion_provider == "fake"
