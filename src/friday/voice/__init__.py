"""Voice subsystem: wake word, audio capture, and voice-activity detection.

This package owns the typed boundaries for FRIDAY's voice pipeline input side
(everything before STT). All real adapters lazy-import their heavy backend
(``openwakeword``, ``sounddevice``) *inside* methods/``__init__`` and raise a
clear error with a ``make install-voice`` hint when the backend is missing, so
importing this package never requires any heavy voice library and ``uv sync``
stays unaffected.

No LLM SDK is imported anywhere in this package (architecture guard).
"""

from __future__ import annotations

from friday.voice.capture import AudioCapture, FakeAudioCapture, MicCapture
from friday.voice.pipeline import StreamingTTS, VoicePipeline, VoiceTurn
from friday.voice.vad import VAD, EnergyVAD, FakeVAD
from friday.voice.wake_word import (
    FakeWakeWord,
    OpenWakeWordDetector,
    WakeResult,
    WakeWordDetector,
)

__all__ = [
    "VAD",
    "AudioCapture",
    "EnergyVAD",
    "FakeAudioCapture",
    "FakeVAD",
    "FakeWakeWord",
    "MicCapture",
    "OpenWakeWordDetector",
    "StreamingTTS",
    "VoicePipeline",
    "VoiceTurn",
    "WakeResult",
    "WakeWordDetector",
]
