"""Opus, via ctypes against the system libopus.

Discord's voice socket speaks Opus and nothing else, in both directions, so the
codec is needed to say anything and to understand anything. There is no usable
pure-Python Opus implementation, and the PyPI bindings are unmaintained wrappers
around the same shared library — so this calls ``libopus`` directly. The image
installs it (see the Dockerfile); this is precisely why the service had to move
off a managed Python runtime, where no system package can be installed.

Only four entry points are needed out of a large C API — create, encode, decode,
destroy — so a hand-rolled binding is smaller than the dependency would be, and
does not add a package that stops being maintained.

Everything here is fixed to Discord's voice format, which is not negotiable at
the far end: **48 kHz, stereo, 20 ms frames**. That makes one frame 960 samples
per channel, which every function below assumes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

from friday.logging import get_logger

logger = get_logger("friday.discord.opus")

#: Discord's voice format. Constants rather than parameters: a mismatch produces
#: audio that is silently wrong — chipmunked or half-speed — rather than an error.
SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_MS = 20
#: Samples *per channel* in one frame: 48000 * 20 / 1000.
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000
#: Bytes in one decoded frame: samples x channels x 2 (16-bit).
FRAME_BYTES = FRAME_SIZE * CHANNELS * 2
#: Generous ceiling for one encoded frame; Opus never approaches it at our bitrate.
MAX_PACKET = 1500

#: ``OPUS_APPLICATION_AUDIO`` — general audio rather than the speech-optimised
#: VOIP mode, because she often reads full sentences and the difference is
#: audible.
_APPLICATION_AUDIO = 2049
#: ``OPUS_SET_BITRATE_REQUEST``. 64 kbit/s is transparent for speech and well
#: inside what Discord accepts.
_SET_BITRATE = 4002
_BITRATE = 64_000


class OpusError(RuntimeError):
    """Raised when libopus is missing or a call fails."""


_lib: Any = None


def _load() -> Any:
    """Find and bind libopus once, or raise :class:`OpusError`."""
    global _lib  # noqa: PLW0603 - a process-wide handle to a shared library
    if _lib is not None:
        return _lib

    found = ctypes.util.find_library("opus")
    # find_library needs ldconfig or gcc, and a slim image has neither. The
    # soname is stable, so try it directly before giving up.
    for candidate in ([found] if found else []) + ["libopus.so.0", "libopus.so"]:
        try:
            lib = ctypes.cdll.LoadLibrary(candidate)
        except OSError:
            continue
        _bind(lib)
        _lib = lib
        logger.info("libopus loaded from %s", candidate)
        return lib
    raise OpusError("libopus not found — install libopus0 (the image does)")


def _bind(lib: Any) -> None:
    """Declare argument and return types.

    Not decoration: without these ctypes assumes ``int`` for every pointer, which
    truncates on 64-bit and corrupts memory in ways that surface far from here.
    """
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.opus_encode.restype = ctypes.c_int32
    lib.opus_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int32,
    ]
    lib.opus_encoder_ctl.restype = ctypes.c_int
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]

    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.opus_decode.restype = ctypes.c_int
    lib.opus_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int,
    ]
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]


def available() -> bool:
    """Whether libopus can be loaded — callers degrade rather than crash."""
    try:
        _load()
    except OpusError:
        return False
    return True


class Encoder:
    """PCM in, Opus frames out. One per voice connection."""

    def __init__(self) -> None:
        lib = _load()
        error = ctypes.c_int()
        self._state = lib.opus_encoder_create(
            SAMPLE_RATE, CHANNELS, _APPLICATION_AUDIO, ctypes.byref(error)
        )
        if error.value != 0 or not self._state:
            raise OpusError(f"opus_encoder_create failed: {error.value}")
        lib.opus_encoder_ctl(self._state, _SET_BITRATE, _BITRATE)
        self._lib = lib

    def encode(self, pcm: bytes) -> bytes:
        """Encode exactly one 20 ms frame of 48 kHz stereo 16-bit PCM."""
        if len(pcm) != FRAME_BYTES:
            raise OpusError(f"expected {FRAME_BYTES} bytes, got {len(pcm)}")
        out = ctypes.create_string_buffer(MAX_PACKET)
        # A mutable buffer, because ctypes cannot take a pointer into immutable
        # bytes. The copy is one frame, 3840 bytes, fifty times a second.
        source = ctypes.create_string_buffer(pcm, len(pcm))
        written = self._lib.opus_encode(
            self._state,
            ctypes.cast(source, ctypes.POINTER(ctypes.c_int16)),
            FRAME_SIZE,
            out,
            MAX_PACKET,
        )
        if written < 0:
            raise OpusError(f"opus_encode failed: {written}")
        return out.raw[:written]

    def close(self) -> None:
        if getattr(self, "_state", None):
            self._lib.opus_encoder_destroy(self._state)
            self._state = None


class Decoder:
    """Opus frames in, PCM out. One per *speaker*, not per connection.

    Opus is stateful: each stream carries prediction state across frames, so
    feeding two people's audio through one decoder produces artefacts. Discord
    identifies each speaker by SSRC, and the caller keeps a decoder per SSRC.
    """

    def __init__(self) -> None:
        lib = _load()
        error = ctypes.c_int()
        self._state = lib.opus_decoder_create(
            SAMPLE_RATE, CHANNELS, ctypes.byref(error)
        )
        if error.value != 0 or not self._state:
            raise OpusError(f"opus_decoder_create failed: {error.value}")
        self._lib = lib

    def decode(self, packet: bytes) -> bytes:
        """Decode one Opus packet to 48 kHz stereo 16-bit PCM."""
        pcm = (ctypes.c_int16 * (FRAME_SIZE * CHANNELS))()
        samples = self._lib.opus_decode(
            self._state, packet, len(packet), pcm, FRAME_SIZE, 0
        )
        if samples < 0:
            raise OpusError(f"opus_decode failed: {samples}")
        return bytes(bytearray(pcm)[: samples * CHANNELS * 2])

    def close(self) -> None:
        if getattr(self, "_state", None):
            self._lib.opus_decoder_destroy(self._state)
            self._state = None
