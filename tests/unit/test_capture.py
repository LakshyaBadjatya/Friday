"""Unit tests for the audio-capture boundary, fake, and lazy mic adapter.

Pins the A2 capture contract: :class:`FakeAudioCapture` replays fixture frames
as an async iterator, and the real :class:`MicCapture` raises a helpful error
when ``sounddevice`` is absent. No real audio device, no network.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from friday.errors import ProviderError
from friday.voice.capture import AudioCapture, FakeAudioCapture, MicCapture
from friday.voice.fixtures import make_silence_frame, make_wake_frame


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fake_is_audio_capture() -> None:
    assert isinstance(FakeAudioCapture([]), AudioCapture)


# --------------------------------------------------------------------------- #
# FakeAudioCapture
# --------------------------------------------------------------------------- #
async def test_fake_yields_frames_in_order() -> None:
    frames = [make_wake_frame(), make_silence_frame(), b"\x00\x01"]
    capture = FakeAudioCapture(frames)
    collected = [frame async for frame in capture.frames()]
    assert collected == frames


async def test_fake_empty_yields_nothing() -> None:
    capture = FakeAudioCapture([])
    collected = [frame async for frame in capture.frames()]
    assert collected == []


async def test_fake_is_independently_replayable() -> None:
    frames = [b"a", b"b"]
    capture = FakeAudioCapture(frames)
    first = [f async for f in capture.frames()]
    second = [f async for f in capture.frames()]
    assert first == second == frames


# --------------------------------------------------------------------------- #
# Real adapter: lazy import, helpful error when backend missing
# --------------------------------------------------------------------------- #
def test_mic_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sounddevice" or name.startswith("sounddevice."):
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError) as exc:
        MicCapture()
    assert "install-voice" in str(exc.value)


def test_module_import_does_not_require_sounddevice() -> None:
    import importlib
    import sys

    assert "sounddevice" not in sys.modules
    importlib.import_module("friday.voice.capture")
    assert "sounddevice" not in sys.modules
