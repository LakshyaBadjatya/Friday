"""Speech-emotion provider boundary, fake, and lazy ONNX adapter.

Emotion is paralinguistic: derived from acoustics, not the transcript. The
representation is dimensional (valence/arousal/dominance) with a derived
categorical label + intensity, so subtle shifts register on a continuous space
rather than collapsing into a few buckets.

No model SDK is imported at module top level: the real :class:`DimEmotion`
adapter lazy-imports ``onnxruntime``/``numpy`` inside its constructor, so
importing this module never pulls in a heavy backend (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.errors import ProviderError


class Emotion(BaseModel):
    """A normalized speech-emotion reading for one window or utterance.

    Attributes:
        valence: Unpleasant (0) -> pleasant (1).
        arousal: Calm (0) -> excited (1).
        dominance: Submissive (0) -> in-control (1).
        label: Nearest categorical name derived from (valence, arousal, dominance).
        intensity: Distance from the neutral centre, 0 (neutral) -> 1 (extreme).
        confidence: Provider confidence in the reading, 0..1.
        ts: Stream timestamp in seconds (monotonic within an analyzer run).
    """

    valence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    label: str
    intensity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    ts: float = 0.0


# Region centroids in (valence, arousal, dominance) space. The nearest centroid
# to a reading gives its label; the distance from the neutral centroid gives the
# intensity. Labels are regions of a continuous space, so "all emotions" is a
# lookup table, not a retrain.
_CENTROIDS: dict[str, tuple[float, float, float]] = {
    "neutral": (0.5, 0.5, 0.5),
    "happy": (0.85, 0.75, 0.6),
    "excited": (0.7, 0.9, 0.6),
    "tender": (0.75, 0.35, 0.45),
    "calm": (0.6, 0.2, 0.5),
    "bored": (0.4, 0.2, 0.4),
    "sad": (0.2, 0.25, 0.3),
    "fearful": (0.25, 0.8, 0.2),
    "angry": (0.2, 0.8, 0.75),
}


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two points in V/A/D space."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def derive_label(valence: float, arousal: float, dominance: float) -> tuple[str, float]:
    """Map a (V, A, D) point to ``(nearest label, intensity in 0..1)``.

    Intensity is the distance from the neutral centroid normalized by the maximum
    reachable distance in the unit cube from its centre (``sqrt(0.75)``), clamped
    to ``1.0``.
    """
    point = (valence, arousal, dominance)
    label = min(_CENTROIDS, key=lambda k: _dist(point, _CENTROIDS[k]))
    intensity = min(1.0, _dist(point, _CENTROIDS["neutral"]) / math.sqrt(0.75))
    return label, intensity


def emotion_hint(emotion: Emotion) -> str:
    """A short, honesty-preserving tone hint for the LLM system prompt (P2).

    The hint nudges tone only; it explicitly forbids asserting the owner's
    feelings as fact or giving clinical/diagnostic readings.
    """
    return (
        f"Voice cue: the owner sounds {emotion.label} "
        f"(intensity {round(emotion.intensity * 100)}%, "
        f"confidence {round(emotion.confidence * 100)}%). "
        "Let this gently shape your tone — warmer and calmer if they seem sad or "
        "afraid, matched energy if they sound upbeat. Do NOT state their emotion "
        "back as fact, do not claim to know how they feel, and never give a "
        "clinical or diagnostic reading."
    )


class EmotionCalibration(BaseModel):
    """Owner personalization (P3): a V/A/D offset recentring the owner's neutral.

    When the owner's *neutral* speech reads as ``m`` on the base model, the offset
    is ``(0.5, 0.5, 0.5) - m``; adding it maps the owner's neutral back to the
    population centre so subtle shifts read relative to *their* baseline.
    """

    v_off: float = 0.0
    a_off: float = 0.0
    d_off: float = 0.0

    def apply(self, emotion: Emotion) -> Emotion:
        """Return ``emotion`` recentred by this calibration (clamped, relabelled)."""
        v = min(1.0, max(0.0, emotion.valence + self.v_off))
        a = min(1.0, max(0.0, emotion.arousal + self.a_off))
        d = min(1.0, max(0.0, emotion.dominance + self.d_off))
        label, intensity = derive_label(v, a, d)
        return emotion.model_copy(update={
            "valence": v, "arousal": a, "dominance": d,
            "label": label, "intensity": intensity,
        })


def calibrate_from_vad(
    neutral_vad: list[tuple[float, float, float]],
) -> EmotionCalibration:
    """Build a calibration so the owner's mean *neutral* reading maps to centre."""
    if not neutral_vad:
        return EmotionCalibration()
    n = len(neutral_vad)
    mv = sum(v for v, _, _ in neutral_vad) / n
    ma = sum(a for _, a, _ in neutral_vad) / n
    md = sum(d for _, _, d in neutral_vad) / n
    return EmotionCalibration(v_off=0.5 - mv, a_off=0.5 - ma, d_off=0.5 - md)


class CalibratedEmotion:
    """Wrap an :class:`EmotionProvider`, applying owner :class:`EmotionCalibration`."""

    def __init__(self, base: EmotionProvider, calibration: EmotionCalibration) -> None:
        self._base = base
        self._cal = calibration

    async def analyze(self, audio: bytes, sr: int = 16000) -> Emotion:
        return self._cal.apply(await self._base.analyze(audio, sr=sr))


async def enroll_owner(
    base: EmotionProvider, neutral_clips: list[bytes], sr: int = 16000
) -> EmotionCalibration:
    """Build an :class:`EmotionCalibration` from the owner's neutral clips."""
    readings = [await base.analyze(c, sr=sr) for c in neutral_clips]
    return calibrate_from_vad([(r.valence, r.arousal, r.dominance) for r in readings])


@runtime_checkable
class EmotionProvider(Protocol):
    """Async contract turning raw 16-bit PCM mono audio into an :class:`Emotion`."""

    async def analyze(self, audio: bytes, sr: int = 16000) -> Emotion:
        """Return the :class:`Emotion` for ``audio`` (16-bit PCM mono at ``sr``)."""
        ...


class FakeEmotion:
    """Deterministic provider for tests: returns a fixed V/A/D (zero models)."""

    def __init__(
        self,
        valence: float = 0.5,
        arousal: float = 0.5,
        dominance: float = 0.5,
        confidence: float = 1.0,
    ) -> None:
        self._v, self._a, self._d, self._c = valence, arousal, dominance, confidence

    async def analyze(self, audio: bytes, sr: int = 16000) -> Emotion:
        label, intensity = derive_label(self._v, self._a, self._d)
        return Emotion(
            valence=self._v, arousal=self._a, dominance=self._d,
            label=label, intensity=intensity, confidence=self._c, ts=0.0,
        )


_EMOTION_INSTALL_HINT = (
    "onnxruntime/numpy are required for DimEmotion, and a V/A/D emotion ONNX "
    "model must be provided via FRIDAY_EMOTION_MODEL (an audeering wav2vec2 "
    "MSP-dim export). Voice extras are optional: `make install-voice`."
)


class DimEmotion:
    """Lazy ONNX adapter: raw 16 kHz waveform -> (arousal, dominance, valence).

    Wraps an audeering ``wav2vec2-large-robust-12-ft-emotion-msp-dim`` ONNX
    export: input ``float32`` waveform shaped ``(1, n_samples)`` at 16 kHz, output
    ``(1, 3)`` = arousal, dominance, valence in [0, 1]. ``onnxruntime``/``numpy``
    are imported lazily in the constructor so importing this module is cheap.
    """

    def __init__(self, model_path: str) -> None:
        import os

        if not os.path.exists(model_path):
            raise ProviderError(f"emotion model not found: {model_path}")
        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError(_EMOTION_INSTALL_HINT) from exc
        self._np = np
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

    async def analyze(self, audio: bytes, sr: int = 16000) -> Emotion:
        np = self._np
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return Emotion(
                valence=0.5, arousal=0.5, dominance=0.5, label="neutral",
                intensity=0.0, confidence=0.0, ts=0.0,
            )
        out = self._sess.run(None, {self._input: x[None, :]})[0].ravel()
        arousal, dominance, valence = (float(np.clip(v, 0.0, 1.0)) for v in out[:3])
        label, intensity = derive_label(valence, arousal, dominance)
        return Emotion(
            valence=valence, arousal=arousal, dominance=dominance,
            label=label, intensity=intensity, confidence=1.0, ts=0.0,
        )


class LiteEmotion:
    """Lazy ONNX adapter over the custom Kaggle-trained V/A/D head (P3).

    Computes the SAME pure-numpy log-mel summary as the training kernel
    (40 mel bands, 25 ms / 10 ms frames -> 162-dim feature), then runs the head
    ONNX (which standardizes the feature, applies the MLP, and clips to [0, 1]) to
    valence/arousal/dominance. Input is 16 kHz 16-bit PCM mono. Keep this front-end
    byte-identical to ``kaggle_kernel_emotion/emotion_train.py``.
    """

    SR = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 40

    def __init__(self, model_path: str) -> None:
        import os

        if not os.path.exists(model_path):
            raise ProviderError(f"emotion model not found: {model_path}")
        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderError(_EMOTION_INSTALL_HINT) from exc
        self._np = np
        self._window = np.hanning(self.N_FFT).astype(np.float32)
        self._mel = self._build_mel(np)
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

    def _build_mel(self, np):  # noqa: ANN001 - np module injected
        sr, n_fft, n_mels, fmin, fmax = self.SR, self.N_FFT, self.N_MELS, 0.0, 8000.0
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
        mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
        mpts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
        bins = np.floor((n_fft + 1) * mel2hz(mpts) / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(1, n_mels + 1):
            left, centre, right = bins[i - 1], bins[i], bins[i + 1]
            for k in range(left, centre):
                if centre > left:
                    fb[i - 1, k] = (k - left) / (centre - left)
            for k in range(centre, right):
                if right > centre:
                    fb[i - 1, k] = (right - k) / (right - centre)
        return fb

    def _features(self, pcm_int16):  # noqa: ANN001 - numpy array
        np = self._np
        x = pcm_int16.astype(np.float32) / 32768.0
        if x.size < self.N_FFT:
            x = np.pad(x, (0, self.N_FFT - x.size))
        n_frames = max(1, 1 + (len(x) - self.N_FFT) // self.HOP)
        frames = np.stack(
            [x[i * self.HOP:i * self.HOP + self.N_FFT] for i in range(n_frames)]
        ) * self._window
        power = (np.abs(np.fft.rfft(frames, n=self.N_FFT, axis=1)) ** 2).astype(np.float32)
        melspec = np.log(power @ self._mel.T + 1e-6)
        rms = np.log(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
        delta = (np.diff(melspec, axis=0) if n_frames > 1
                 else np.zeros((1, self.N_MELS), np.float32))
        return np.concatenate([
            melspec.mean(0), melspec.std(0), delta.mean(0), delta.std(0),
            [rms.mean(), rms.std()],
        ]).astype(np.float32)

    async def analyze(self, audio: bytes, sr: int = 16000) -> Emotion:
        np = self._np
        pcm = np.frombuffer(audio, dtype=np.int16)
        if pcm.size == 0:
            return Emotion(valence=0.5, arousal=0.5, dominance=0.5, label="neutral",
                           intensity=0.0, confidence=0.0, ts=0.0)
        feat = self._features(pcm)[None, :]
        out = self._sess.run(None, {self._input: feat})[0].ravel()
        valence, arousal, dominance = (float(min(1.0, max(0.0, v))) for v in out[:3])
        label, intensity = derive_label(valence, arousal, dominance)
        return Emotion(valence=valence, arousal=arousal, dominance=dominance,
                       label=label, intensity=intensity, confidence=1.0, ts=0.0)
