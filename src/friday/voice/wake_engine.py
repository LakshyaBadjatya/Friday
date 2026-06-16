# © Lakshya Badjatya — Author
"""The wake-word engine seam: turn an audio frame into a 'hey friday' score.

Offline-first, exactly like the rest of the voice/perception stack: a deterministic
:class:`FakeWakeWordEngine` backs the tests and the no-deps default build, while the
real :class:`OpenWakeWordEngine` lazy-imports ``openwakeword`` and loads the trained
ONNX model (``FRIDAY_WAKEWORD_MODEL``, produced by ``notebooks/train_hey_friday_wakeword.ipynb``)
only when wake-word detection is enabled. Neither the optional ``openwakeword`` nor
``numpy`` import happens at module import time, so this module is always importable
and the gate stays green without the voice extras.

This module only scores a frame and answers "did it cross the threshold?" — the
mic-capture loop and the WebSocket push to the HUD live in the app layer; the
summon phrase ("FRIDAY summon VISION") is parsed by :func:`friday.voice.wake.parse_wake_command`
over an STT transcript once the wake word has fired.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from friday.errors import ProviderError

#: Default detection threshold; a score at or above this is a wake.
DEFAULT_WAKE_THRESHOLD = 0.5

# openWakeWord's expected frame: 16 kHz mono int16, ~80 ms (1280 samples).
_INSTALL_HINT = (
    "openwakeword is not installed. Install the voice extras "
    "(`make install-voice`) and train a model with "
    "notebooks/train_hey_friday_wakeword.ipynb."
)


@runtime_checkable
class WakeWordEngine(Protocol):
    """Scores one audio frame for the presence of the wake word."""

    def score(self, frame: bytes) -> float:
        """Return a detection score in ``[0, 1]`` for ``frame`` (16 kHz int16 PCM)."""
        ...


class FakeWakeWordEngine:
    """A scripted, deterministic engine for tests and the offline default.

    Pops queued scores in order (defaulting to ``0.0`` once exhausted), so a test
    can drive the wake/no-wake decision without any audio backend.
    """

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores: list[float] = list(scores or [])

    def score(self, frame: bytes) -> float:
        """Return the next scripted score, or ``0.0`` when the script is empty."""
        return self._scores.pop(0) if self._scores else 0.0


def _load_wakeword_model(model_path: str) -> Any:
    """Lazy-load an openWakeWord model; raise a clear error if the extra is absent."""
    try:
        from openwakeword.model import Model  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise ProviderError(_INSTALL_HINT) from exc
    return Model(wakeword_models=[model_path])


class OpenWakeWordEngine:
    """Real engine over ``openwakeword`` + a trained ONNX model (lazy / opt-in).

    The model and ``numpy`` are imported on first :meth:`score`, so constructing
    this is cheap and import-safe even without the extras; a missing dependency
    surfaces as a clear :class:`~friday.errors.ProviderError`.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_wakeword_model(self._model_path)
        return self._model

    def score(self, frame: bytes) -> float:
        """Score ``frame`` via openWakeWord; the max over the loaded model(s)."""
        try:
            import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - numpy ships with the extras
            raise ProviderError(_INSTALL_HINT) from exc
        model = self._ensure_model()
        audio = np.frombuffer(frame, dtype=np.int16)
        predictions = model.predict(audio)
        return float(max(predictions.values())) if predictions else 0.0


def detected(score: float, threshold: float = DEFAULT_WAKE_THRESHOLD) -> bool:
    """Whether a wake-word ``score`` crosses ``threshold`` (boundary-inclusive)."""
    return score >= threshold
