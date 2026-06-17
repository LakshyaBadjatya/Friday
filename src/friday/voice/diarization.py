# © Lakshya Badjatya — Author
"""Speaker diarization seam — "who spoke when" over an audio clip.

Mirrors the other optional-backend seams: a runtime-checkable :class:`Diarizer`
protocol, a deterministic :class:`FakeDiarizer` for tests / offline builds, and a
lazily-imported :class:`PyannoteDiarizer` that touches ``pyannote.audio`` only
inside its method. :func:`build_diarizer` returns ``None`` unless
``enable_diarization`` is on, and falls back to the fake when the heavy backend
is not installed — so enabling the flag never crashes a headless box.

Diarization output (a list of :class:`SpeakerSegment`) is what a meeting-capture
pipeline overlays on a transcript to attribute lines to speakers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from friday.config import Settings

logger = logging.getLogger("friday.voice.diarization")


class SpeakerSegment(BaseModel):
    """One contiguous span attributed to a speaker (seconds, half-open)."""

    speaker: str
    start: float
    end: float


@runtime_checkable
class Diarizer(Protocol):
    """Anything that can segment an audio file by speaker."""

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """Return speaker segments for the audio at ``audio_path``."""
        ...


class FakeDiarizer:
    """A deterministic diarizer that ignores the audio and returns canned turns.

    Defaults to an alternating two-speaker pattern; pass ``segments`` to script an
    exact result in tests. Lets the diarization surface be exercised with no audio
    backend and no real audio file.
    """

    def __init__(
        self,
        segments: list[SpeakerSegment] | None = None,
        *,
        speakers: int = 2,
        span: float = 4.0,
        turns: int = 4,
    ) -> None:
        if segments is not None:
            self._segments = list(segments)
        else:
            self._segments = [
                SpeakerSegment(
                    speaker=f"SPEAKER_{i % speakers:02d}",
                    start=i * span,
                    end=(i + 1) * span,
                )
                for i in range(turns)
            ]

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """Return the canned segmentation, regardless of ``audio_path``."""
        return list(self._segments)


class PyannoteDiarizer:
    """Real diarizer over ``pyannote.audio`` (lazy-imported in :meth:`diarize`)."""

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1") -> None:
        self._model = model

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """Run the pyannote pipeline and map its turns to :class:`SpeakerSegment`."""
        pipeline = _load_pyannote(self._model)
        annotation = pipeline(audio_path)
        return [
            SpeakerSegment(
                speaker=str(speaker), start=float(turn.start), end=float(turn.end)
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def _load_pyannote(model: str) -> Any:
    """Lazy-load the pyannote pipeline; raises ImportError when not installed."""
    from pyannote.audio import Pipeline  # type: ignore[import-not-found]  # noqa: PLC0415

    return Pipeline.from_pretrained(model)


def _pyannote_available() -> bool:
    """Whether ``pyannote.audio`` can be imported (no model download)."""
    try:
        import pyannote.audio  # type: ignore[import-not-found]  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def build_diarizer(settings: Settings) -> Diarizer | None:
    """Return a diarizer when ``enable_diarization`` is on, else ``None``.

    Uses the real pyannote backend when installed, otherwise the deterministic
    fake (logged) — so the flag is safe to set on a build without the heavy dep.
    """
    if not settings.enable_diarization:
        return None
    if _pyannote_available():
        return PyannoteDiarizer()
    logger.warning(
        "enable_diarization is set but pyannote.audio is not installed; using the "
        "deterministic FakeDiarizer (install the voice extras for real diarization)"
    )
    return FakeDiarizer()
