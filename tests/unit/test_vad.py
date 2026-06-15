"""Unit tests for voice-activity detection: energy detector and fake.

Pins the A2 VAD contract: :class:`EnergyVAD` flags a high-energy tone frame as
speech and a silent frame as non-speech, and :class:`FakeVAD` replays a scripted
boolean sequence. Standard-library audio fixtures only.
"""

from __future__ import annotations

from friday.voice.fixtures import make_silence_frame, make_tone_frame
from friday.voice.vad import VAD, EnergyVAD, FakeVAD


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_energy_vad_is_vad() -> None:
    assert isinstance(EnergyVAD(), VAD)


def test_fake_vad_is_vad() -> None:
    assert isinstance(FakeVAD([]), VAD)


# --------------------------------------------------------------------------- #
# EnergyVAD
# --------------------------------------------------------------------------- #
def test_energy_vad_detects_tone_as_speech() -> None:
    vad = EnergyVAD()
    assert vad.is_speech(make_tone_frame()) is True


def test_energy_vad_treats_silence_as_non_speech() -> None:
    vad = EnergyVAD()
    assert vad.is_speech(make_silence_frame()) is False


def test_energy_vad_empty_frame_is_non_speech() -> None:
    vad = EnergyVAD()
    assert vad.is_speech(b"") is False


def test_energy_vad_threshold_is_configurable() -> None:
    # A threshold above the tone's RMS flips the tone to non-speech.
    loud = make_tone_frame()
    permissive = EnergyVAD(threshold=1.0)
    strict = EnergyVAD(threshold=1_000_000.0)
    assert permissive.is_speech(loud) is True
    assert strict.is_speech(loud) is False


def test_energy_vad_tolerates_odd_length_frame() -> None:
    # A trailing odd byte must not raise; it is trimmed before the int16 view.
    vad = EnergyVAD()
    assert vad.is_speech(make_tone_frame() + b"\x01") is True


# --------------------------------------------------------------------------- #
# FakeVAD
# --------------------------------------------------------------------------- #
def test_fake_vad_replays_script() -> None:
    vad = FakeVAD([True, False, True])
    assert [vad.is_speech(b"") for _ in range(3)] == [True, False, True]


def test_fake_vad_returns_false_past_end_of_script() -> None:
    vad = FakeVAD([True])
    assert vad.is_speech(b"") is True
    # Exhausted script -> silence, so polling loops terminate.
    assert vad.is_speech(b"") is False
    assert vad.is_speech(b"") is False
