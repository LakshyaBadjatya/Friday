# © Lakshya Badjatya — Author
"""Unit tests for the speaker-diarization seam (flag-gated, lazy backend)."""

from __future__ import annotations

from friday.config import Settings
from friday.voice.diarization import (
    Diarizer,
    FakeDiarizer,
    SpeakerSegment,
    build_diarizer,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", **overrides)  # type: ignore[arg-type]


def test_fake_diarizer_default_pattern() -> None:
    segs = FakeDiarizer().diarize("anything.wav")
    assert len(segs) == 4
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"
    assert segs[0].end == segs[1].start  # contiguous turns


def test_fake_diarizer_accepts_scripted_segments() -> None:
    canned = [SpeakerSegment(speaker="A", start=0.0, end=1.5)]
    assert FakeDiarizer(canned).diarize("x") == canned


def test_fake_diarizer_satisfies_protocol() -> None:
    assert isinstance(FakeDiarizer(), Diarizer)


def test_build_diarizer_none_when_off() -> None:
    assert build_diarizer(_settings()) is None


def test_build_diarizer_degrades_to_fake_when_backend_missing() -> None:
    # pyannote.audio is not installed here -> enabled build falls back to the fake.
    assert isinstance(build_diarizer(_settings(enable_diarization=True)), FakeDiarizer)
