"""Sliding-window emotion analysis over a capture stream.

:class:`EmotionStreamAnalyzer` buffers raw 16-bit PCM frames into a window, calls
an :class:`~friday.providers.emotion.EmotionProvider` once per hop, EMA-smooths
the valence/arousal/dominance so the HUD reading is stable rather than jittery,
and emits the smoothed :class:`~friday.providers.emotion.Emotion` to registered
listeners. It depends only on the provider boundary — no model is imported here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from friday.providers.emotion import Emotion, EmotionProvider, derive_label

if TYPE_CHECKING:
    from friday.voice.capture import AudioCapture
    from friday.voice.voiceprint import OwnerIdentity

logger = logging.getLogger(__name__)

EmotionListener = Callable[[Emotion], None]


async def feed_analyzer(
    capture: AudioCapture, analyzer: EmotionStreamAnalyzer
) -> None:
    """Pump frames from ``capture`` into ``analyzer`` until the stream ends.

    The continuous-sensing loop: each captured PCM frame is pushed to the
    analyzer, which emits smoothed Emotions to its listeners (e.g. a ``/ws/emotion``
    broadcast). Runs until the capture stream is exhausted or the task is
    cancelled (cooperatively, at the next ``await``).
    """
    async for frame in capture.frames():
        await analyzer.push(frame)


class EmotionStreamAnalyzer:
    """Window + hop + EMA over a PCM frame stream, emitting smoothed Emotions."""

    def __init__(
        self,
        provider: EmotionProvider,
        sr: int = 16000,
        window_s: float = 1.5,
        hop_s: float = 0.5,
        alpha: float = 0.4,
        owner: OwnerIdentity | None = None,
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

    def off_emotion(self, listener: EmotionListener) -> None:
        """Detach a previously registered listener (no-op if not present).

        Callers with a finite lifetime (e.g. a ``/ws/emotion`` connection) must
        detach on teardown so the shared analyzer does not accumulate dead
        listeners — and keep firing into their orphaned queues — forever.
        """
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

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
            ema: tuple[float, float, float] = cur
        else:
            a = self._alpha
            p = self._ema
            ema = (
                a * cur[0] + (1 - a) * p[0],
                a * cur[1] + (1 - a) * p[1],
                a * cur[2] + (1 - a) * p[2],
            )
        self._ema = ema
        valence, arousal, dominance = ema
        label, intensity = derive_label(valence, arousal, dominance)
        self._t += self._hop_s
        emotion = Emotion(
            valence=valence, arousal=arousal, dominance=dominance, label=label,
            intensity=intensity, confidence=raw.confidence, ts=self._t,
        )
        self._last = emotion
        for listener in self._listeners:
            # Isolate listener failures: a single misbehaving listener (e.g. a
            # broken /ws/emotion broadcast) must not propagate out through push()
            # and silently kill the fire-and-forget continuous-sensing task.
            try:
                listener(emotion)
            except Exception:
                logger.exception("emotion listener failed; continuing")
