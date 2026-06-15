"""Deterministic synthetic audio fixtures for voice tests.

Everything here uses only the Python standard library (``wave``, ``struct``,
``math``) so tests never touch a real microphone, file, or network. The bytes
produced are mono 16-bit little-endian PCM at 16 kHz — the canonical format the
voice pipeline expects.

Two frame helpers drive the deterministic fakes:

* :func:`make_wake_frame` embeds a recognizable marker prefix so
  :class:`friday.voice.wake_word.FakeWakeWord` can fire on it and only it.
* :func:`make_silence_frame` returns near-silent PCM that the energy VAD and the
  fake wake word both treat as "no speech / no wake".
"""

from __future__ import annotations

import math
import struct
import wave
from io import BytesIO

# Canonical PCM parameters for the voice pipeline.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)
CHANNELS = 1

# Marker prefix embedded at the start of a positive wake fixture frame. It is
# not valid as the leading bytes of natural 16-bit PCM speech for our fakes, so
# its presence is a deterministic, unambiguous "wake" signal in tests.
WAKE_MARKER = b"WAKE"


def make_wav(seconds: float = 0.2, freq: int = 440, sample_rate: int = 16000) -> bytes:
    """Return a mono 16-bit PCM WAV container as bytes.

    A pure sine tone at ``freq`` Hz lasting ``seconds`` seconds, written through
    the stdlib :mod:`wave` module so the result is a real, parseable WAV file.

    Args:
        seconds: Duration of the tone in seconds.
        freq: Sine frequency in Hz.
        sample_rate: Samples per second.

    Returns:
        The full ``.wav`` file contents (RIFF header + PCM data) as bytes.
    """
    n_samples = int(seconds * sample_rate)
    amplitude = 16000  # comfortably below the int16 ceiling (32767)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames += struct.pack("<h", value)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def make_tone_frame(samples: int = 1600, freq: int = 440) -> bytes:
    """Return raw (headerless) 16-bit PCM bytes of a sine tone.

    Used as a non-wake, high-energy "speech" frame for VAD tests.

    Args:
        samples: Number of PCM samples to generate.
        freq: Sine frequency in Hz.

    Returns:
        Raw little-endian 16-bit mono PCM bytes (no WAV header).
    """
    amplitude = 16000
    out = bytearray()
    for i in range(samples):
        value = int(amplitude * math.sin(2.0 * math.pi * freq * i / SAMPLE_RATE))
        out += struct.pack("<h", value)
    return bytes(out)


def make_silence_frame(samples: int = 1600) -> bytes:
    """Return raw 16-bit PCM bytes of near silence (all-zero samples)."""
    return struct.pack(f"<{samples}h", *([0] * samples))


def make_wake_frame(samples: int = 1600, freq: int = 440) -> bytes:
    """Return a positive wake-word fixture frame.

    The frame is a normal tone frame prefixed with :data:`WAKE_MARKER` so the
    deterministic :class:`friday.voice.wake_word.FakeWakeWord` fires on it (and
    only on frames carrying the marker).

    Args:
        samples: Number of PCM samples in the tone body.
        freq: Sine frequency in Hz for the tone body.

    Returns:
        ``WAKE_MARKER`` followed by raw PCM tone bytes.
    """
    return WAKE_MARKER + make_tone_frame(samples=samples, freq=freq)
