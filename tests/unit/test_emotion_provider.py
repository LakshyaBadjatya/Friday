"""Unit tests for the speech-emotion provider boundary (Phase 1).

Everything here is model-free: ``derive_label`` is a pure function and
``FakeEmotion`` returns a fixed reading, so the suite stays green without any
ONNX model or heavy dependency.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from friday.errors import ProviderError
from friday.providers.emotion import (
    DimEmotion,
    Emotion,
    EmotionProvider,
    FakeEmotion,
    derive_label,
)


def test_derive_label_neutral_centre() -> None:
    # centre of V/A/D space -> neutral, ~zero intensity
    label, intensity = derive_label(0.5, 0.5, 0.5)
    assert label == "neutral"
    assert intensity < 0.1


def test_derive_label_happy_high_valence_high_arousal() -> None:
    label, _ = derive_label(0.9, 0.8, 0.6)
    assert label == "happy"


def test_derive_label_sad_low_valence_low_arousal() -> None:
    label, intensity = derive_label(0.1, 0.2, 0.3)
    assert label == "sad"
    assert intensity > 0.4


def test_derive_label_fearful_low_valence_high_arousal_low_dominance() -> None:
    label, _ = derive_label(0.2, 0.85, 0.2)
    assert label == "fearful"


def test_emotion_model_roundtrips() -> None:
    e = Emotion(
        valence=0.3, arousal=0.6, dominance=0.4, label="sad",
        intensity=0.5, confidence=0.7, ts=1.5,
    )
    assert Emotion.model_validate_json(e.model_dump_json()) == e


def test_fakeemotion_is_provider_and_deterministic() -> None:
    fake = FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4)
    assert isinstance(fake, EmotionProvider)
    a = asyncio.run(fake.analyze(b"\x00\x01" * 8000))
    b = asyncio.run(fake.analyze(b"\x00\x01" * 8000))
    assert a == b
    assert a.label == "sad"  # (0.2, 0.3, 0.4) -> sad via derive_label
    assert 0.0 <= a.confidence <= 1.0


def test_dimemotion_missing_model_raises_providererror() -> None:
    with pytest.raises(ProviderError):
        DimEmotion(model_path="/nonexistent/emotion.onnx")


@pytest.mark.skipif(
    not os.environ.get("FRIDAY_EMOTION_MODEL"),
    reason="no emotion ONNX model provided",
)
def test_dimemotion_smoke() -> None:
    import numpy as np

    prov = DimEmotion(model_path=os.environ["FRIDAY_EMOTION_MODEL"])
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    e = asyncio.run(prov.analyze(pcm))
    assert 0.0 <= e.valence <= 1.0 and 0.0 <= e.arousal <= 1.0
