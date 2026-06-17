"""Sliding-window emotion analysis over a capture stream.

:class:`EmotionStreamAnalyzer` buffers raw 16-bit PCM frames into a window, calls
an :class:`~friday.providers.emotion.EmotionProvider` once per hop, EMA-smooths
the valence/arousal/dominance so the HUD reading is stable rather than jittery,
and emits the smoothed :class:`~friday.providers.emotion.Emotion` to registered
listeners. It depends only on the provider boundary — no model is imported here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from friday.providers.emotion import Emotion, EmotionProvider, derive_label

if TYPE_CHECKING:
    from friday.voice.voiceprint import OwnerIdentity

EmotionListener = Callable[[Emotion], None]


class EmotionStreamAnalyzer:
    """Window + hop + EMA over a PCM frame stream, emitting smoothed Emotions."""

    def __init__(
        self,
        provider: EmotionProvider,
        sr: int = 16000,
        window_s: float = 1.5,
        hop_s: float = 0.5,
        alpha: float = 0.4,
        owner: "OwnerIdentity | None" = None,
        owner_only: bool = False,
    ) -> None:
        self._provider = provider
        self._sr = sr
        self._owner = owner
        self._owner_only = owner_only
        self._bytes_per_sample = 2  # 16-bit PCM mono
        self._window_bytes = int(window_s * sr) * self._bytes_per_sample
        self._hop_bytes = int(hop_s * sr) * self._bytes_per_sample
        self._alpha = alpha
        self._hop_s = hop_s
        self._buf = bytearray()
        self._since_hop = 0
        self._ema: tuple[float, float, float] | None = None
        self._last: Emotion | None = None
        self._listeners: list[EmotionListener] = []
        self._t = 0.0

    def on_emotion(self, listener: EmotionListener) -> None:
        """Register ``listener`` to be called on every smoothed Emotion."""
        self._listeners.append(listener)

    def last(self) -> Emotion | None:
        """The most recent smoothed Emotion, or ``None`` before the first hop."""
        return self._last

    async def push(self, frame: bytes) -> None:
        """Feed one capture frame; emit a smoothed Emotion once a hop accumulates."""
        self._buf.extend(frame)
        self._since_hop += len(frame)
        if len(self._buf) > self._window_bytes:
            del self._buf[: len(self._buf) - self._window_bytes]
        while self._since_hop >= self._hop_bytes and len(self._buf) >= self._hop_bytes:
            self._since_hop -= self._hop_bytes
            await self._emit()

    async def _emit(self) -> None:
        window = bytes(self._buf)
        # Owner-gating (advisory): when enabled, only the owner's voice drives the
        # signal — a non-owner window is skipped (no emit, last() unchanged).
        if self._owner_only and self._owner is not None and not self._owner.is_owner(window):
            return
        raw = await self._provider.analyze(window, sr=self._sr)
        cur = (raw.valence, raw.arousal, raw.dominance)
        if self._ema is None:
            self._ema = cur
        else:
            a = self._alpha
            self._ema = tuple(
                a * c + (1 - a) * p for c, p in zip(cur, self._ema, strict=True)
            )  # type: ignore[assignment]
        valence, arousal, dominance = self._ema
        label, intensity = derive_label(valence, arousal, dominance)
        self._t += self._hop_s
        emotion = Emotion(
            valence=valence, arousal=arousal, dominance=dominance, label=label,
            intensity=intensity, confidence=raw.confidence, ts=self._t,
        )
        self._last = emotion
        for listener in self._listeners:
            listener(emotion)
