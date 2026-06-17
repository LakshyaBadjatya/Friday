"""Phase-3A: owner personalization (V/A/D recentring). Model-free."""

from __future__ import annotations

import asyncio

from friday.providers.emotion import (
    CalibratedEmotion,
    Emotion,
    EmotionCalibration,
    EmotionProvider,
    FakeEmotion,
    calibrate_from_vad,
    derive_label,
    enroll_owner,
)


def _e(v: float, a: float, d: float) -> Emotion:
    label, intensity = derive_label(v, a, d)
    return Emotion(valence=v, arousal=a, dominance=d, label=label,
                   intensity=intensity, confidence=1.0, ts=0.0)


def test_calibration_recenters_owner_neutral() -> None:
    # Owner's "neutral" reads as (0.3, 0.6, 0.5) on the base model.
    cal = calibrate_from_vad([(0.3, 0.6, 0.5)])
    out = cal.apply(_e(0.3, 0.6, 0.5))
    assert abs(out.valence - 0.5) < 1e-6 and abs(out.arousal - 0.5) < 1e-6
    assert out.label == "neutral"


def test_calibration_clamps_and_roundtrips() -> None:
    cal = calibrate_from_vad([(0.1, 0.1, 0.1)])  # large positive offset
    out = cal.apply(_e(0.9, 0.9, 0.9))
    assert 0.0 <= out.valence <= 1.0 and 0.0 <= out.arousal <= 1.0
    assert EmotionCalibration.model_validate_json(cal.model_dump_json()) == cal


def test_calibrated_provider_recenters() -> None:
    base = FakeEmotion(valence=0.3, arousal=0.6, dominance=0.5)
    cal = asyncio.run(enroll_owner(base, [b"\x00\x01" * 8000]))
    prov = CalibratedEmotion(base, cal)
    assert isinstance(prov, EmotionProvider)
    out = asyncio.run(prov.analyze(b"\x00\x01" * 8000))
    assert abs(out.valence - 0.5) < 1e-6 and out.label == "neutral"
