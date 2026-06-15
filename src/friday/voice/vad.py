"""Voice-activity detection (VAD) boundary, energy detector, and fake.

* :class:`VAD` — the runtime-checkable ``is_speech`` protocol.
* :class:`EnergyVAD` — a dependency-free detector that flags a frame as speech
  when its RMS energy over 16-bit PCM exceeds a threshold.
* :class:`FakeVAD` — replays a scripted boolean sequence for deterministic tests.

Only the standard library is used here; no heavy voice backend is imported.
"""

from __future__ import annotations

import array
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# RMS energy (in int16 amplitude units) above which a frame counts as speech.
DEFAULT_ENERGY_THRESHOLD = 500.0


@runtime_checkable
class VAD(Protocol):
    """Contract deciding whether an audio ``frame`` contains speech."""

    def is_speech(self, frame: bytes) -> bool:
        """Return ``True`` if ``frame`` is judged to contain speech."""
        ...


class EnergyVAD:
    """An RMS-energy :class:`VAD` over 16-bit little-endian PCM.

    A frame counts as speech when its root-mean-square amplitude exceeds
    ``threshold``. Silence (all-zero or low-amplitude PCM) falls below it.
    """

    def __init__(self, threshold: float = DEFAULT_ENERGY_THRESHOLD) -> None:
        """Create the detector.

        Args:
            threshold: RMS amplitude (int16 units) above which a frame is speech.
        """
        self.threshold = float(threshold)

    def is_speech(self, frame: bytes) -> bool:
        rms = self._rms(frame)
        return rms > self.threshold

    @staticmethod
    def _rms(frame: bytes) -> float:
        """Root-mean-square amplitude of a 16-bit little-endian PCM frame."""
        # Trim any trailing odd byte so the int16 view is well-formed.
        usable = frame[: len(frame) - (len(frame) % 2)]
        if not usable:
            return 0.0
        samples = array.array("h")
        samples.frombytes(usable)
        if not samples:
            return 0.0
        total = sum(sample * sample for sample in samples)
        return math.sqrt(total / len(samples))


class FakeVAD:
    """A deterministic :class:`VAD` replaying a scripted boolean sequence.

    Each :meth:`is_speech` call returns the next value from the script; once the
    script is exhausted it returns ``False`` (treated as silence) so callers that
    keep polling terminate naturally.
    """

    def __init__(self, script: Sequence[bool]) -> None:
        """Store the scripted decisions.

        Args:
            script: The boolean results to return, in order.
        """
        self._script = list(script)
        self._index = 0

    def is_speech(self, frame: bytes) -> bool:
        if self._index >= len(self._script):
            return False
        value = self._script[self._index]
        self._index += 1
        return value
