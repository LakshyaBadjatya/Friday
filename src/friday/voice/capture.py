"""Audio capture boundary, deterministic fake, and lazy real microphone adapter.

* :class:`AudioCapture` — the runtime-checkable async-iterator ``frames`` protocol.
* :class:`FakeAudioCapture` — yields a fixed list of fixture frames (no device).
* :class:`MicCapture` — the real adapter that lazy-imports ``sounddevice`` and
  raises a clear error (with a ``make install-voice`` hint) when it is absent.

No heavy audio library is imported at module top level.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from friday.errors import ProviderError
from friday.voice.fixtures import CHANNELS, SAMPLE_RATE

_INSTALL_HINT = (
    "sounddevice is not installed. Voice extras are optional and excluded from "
    "the uv lock; install them with `make install-voice`."
)


@runtime_checkable
class AudioCapture(Protocol):
    """Contract producing an async stream of raw PCM audio frames."""

    def frames(self) -> AsyncIterator[bytes]:
        """Return an async iterator yielding raw PCM frames as they arrive."""
        ...


class FakeAudioCapture:
    """A deterministic :class:`AudioCapture` for tests.

    Yields the frames it was constructed with, in order, then stops — no real
    audio device, no blocking, no network.
    """

    def __init__(self, frames: Sequence[bytes]) -> None:
        """Store the fixture frames to replay.

        Args:
            frames: The PCM frames to yield, in order.
        """
        self._frames = list(frames)

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame


class MicCapture:
    """Real :class:`AudioCapture` backed by ``sounddevice`` (lazy).

    The ``sounddevice`` import happens inside ``__init__`` so importing this
    module never requires the backend. When it is missing, a
    :class:`friday.errors.ProviderError` is raised with an install hint.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        block_size: int = 1600,
    ) -> None:
        """Open the microphone stream parameters and load ``sounddevice``.

        Args:
            sample_rate: Capture sample rate in Hz.
            channels: Number of input channels.
            block_size: Frames per read block.

        Raises:
            ProviderError: If ``sounddevice`` is not installed.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        try:
            # Optional voice backend: excluded from the uv lock, so mypy has no
            # stub for it; lazily imported here and guarded by the ImportError.
            import sounddevice  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        self._sounddevice = sounddevice

    async def frames(self) -> AsyncIterator[bytes]:  # pragma: no cover - needs a mic
        """Yield raw 16-bit PCM blocks read from the default input device."""
        with self._sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
        ) as stream:
            while True:
                data, _overflowed = stream.read(self.block_size)
                yield bytes(data)
