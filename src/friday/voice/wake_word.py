"""Wake-word detection boundary, deterministic fake, and lazy real adapter.

* :class:`WakeResult` — the pydantic v2 detection result.
* :class:`WakeWordDetector` — the runtime-checkable ``detect`` protocol.
* :class:`FakeWakeWord` — deterministic detector for tests: fires (score above
  threshold) only on a frame produced by
  :func:`friday.voice.fixtures.make_wake_frame`.
* :class:`OpenWakeWordDetector` — the real adapter that lazy-imports
  ``openwakeword`` inside ``__init__`` and raises a clear error (with a
  ``make install-voice`` hint) when the backend is absent.

The default detection threshold comes from application settings
(``FRIDAY_WAKE_WORD_THRESHOLD`` when present) and falls back to
:data:`DEFAULT_WAKE_THRESHOLD`. No heavy voice library is imported at module
top level.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.config import get_settings
from friday.errors import ProviderError
from friday.voice.fixtures import WAKE_MARKER

# Fallback wake-word confidence threshold when settings does not define one.
DEFAULT_WAKE_THRESHOLD = 0.5

_INSTALL_HINT = (
    "openwakeword is not installed. Voice extras are optional and excluded from "
    "the uv lock; install them with `make install-voice`."
)


def _settings_threshold() -> float:
    """Return the configured wake threshold, or :data:`DEFAULT_WAKE_THRESHOLD`.

    Reads ``wake_word_threshold`` off :func:`friday.config.get_settings` when the
    field exists, keeping this slice independent of whether the config field has
    landed yet.
    """
    raw = getattr(get_settings(), "wake_word_threshold", DEFAULT_WAKE_THRESHOLD)
    return float(raw)


class WakeResult(BaseModel):
    """Result of evaluating a single audio frame for the wake word.

    Attributes:
        detected: Whether the wake word was detected in the frame.
        score: Detection confidence in ``[0.0, 1.0]``.
    """

    detected: bool
    score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class WakeWordDetector(Protocol):
    """Contract evaluating an audio ``frame`` for a wake word."""

    def detect(self, frame: bytes) -> WakeResult:
        """Evaluate ``frame`` and return a :class:`WakeResult`.

        Args:
            frame: Raw PCM audio bytes for a single frame.

        Returns:
            The :class:`WakeResult` for this frame.
        """
        ...


class FakeWakeWord:
    """A deterministic :class:`WakeWordDetector` for tests.

    Fires (``detected=True`` with ``score >= threshold``) exactly when ``frame``
    begins with :data:`friday.voice.fixtures.WAKE_MARKER` — i.e. a frame produced
    by :func:`friday.voice.fixtures.make_wake_frame`. Any other (plain/negative)
    frame yields ``detected=False`` with a score strictly *below* threshold.
    """

    def __init__(self, threshold: float | None = None) -> None:
        """Create the fake detector.

        Args:
            threshold: Detection threshold; defaults to the configured wake
                threshold (or :data:`DEFAULT_WAKE_THRESHOLD`).
        """
        self.threshold = _settings_threshold() if threshold is None else float(threshold)

    def detect(self, frame: bytes) -> WakeResult:
        if frame.startswith(WAKE_MARKER):
            # A positive fixture: confidently above the threshold boundary.
            return WakeResult(detected=True, score=1.0)
        # A negative frame: strictly below the threshold boundary.
        return WakeResult(detected=False, score=0.0)


class OpenWakeWordDetector:
    """Real :class:`WakeWordDetector` backed by ``openwakeword`` (lazy).

    The heavy ``openwakeword`` import happens inside ``__init__`` so importing
    this module never requires the backend. When the backend is missing, a
    :class:`friday.errors.ProviderError` is raised with an install hint.
    """

    def __init__(self, model_name: str | None = None, threshold: float | None = None) -> None:
        """Construct the detector, loading the ``openwakeword`` model.

        Args:
            model_name: Optional model name passed to ``openwakeword``; ``None``
                uses the backend default.
            threshold: Detection threshold; defaults to the configured wake
                threshold (or :data:`DEFAULT_WAKE_THRESHOLD`).

        Raises:
            ProviderError: If ``openwakeword`` is not installed.
        """
        self.threshold = _settings_threshold() if threshold is None else float(threshold)
        try:
            # Optional voice backend: excluded from the uv lock, so mypy has no
            # stub for it; lazily imported here and guarded by the ImportError.
            import openwakeword  # type: ignore[import-not-found]  # noqa: PLC0415
            from openwakeword.model import Model  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        self._openwakeword = openwakeword
        models = [model_name] if model_name else []
        self._model = Model(wakeword_models=models) if models else Model()

    def detect(self, frame: bytes) -> WakeResult:
        """Evaluate ``frame`` with the loaded model and return the best score."""
        import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415

        # 16-bit PCM needs an even byte count; np.frombuffer raises ValueError on
        # a stray odd-length frame (e.g. a truncated capture chunk). Drop the
        # trailing odd byte so a malformed frame degrades to a slightly-short
        # window rather than crashing the wake detector.
        if len(frame) % 2:
            frame = frame[:-1]
        audio = np.frombuffer(frame, dtype=np.int16)
        scores = self._model.predict(audio)
        best = max(scores.values()) if scores else 0.0
        score = float(best)
        return WakeResult(detected=score >= self.threshold, score=score)
