"""The voice pipeline: wake -> capture -> VAD -> STT -> orchestrator -> TTS.

:class:`VoicePipeline` drives one spoken turn end-to-end and emits the core
:class:`~friday.core.state.Mode` transitions (``IDLE`` -> ``LISTENING`` ->
``ROUTING``) as it advances, so a UI or logger can observe state without polling.

``run_once`` waits for a wake detection over the capture stream, records the
utterance until the VAD reports silence, transcribes it, runs the orchestrator,
and synthesizes the reply to audio bytes — returning a :class:`VoiceTurn` (or
``None`` if the capture stream ends before the wake word fires).

Barge-in (``speak_with_bargein``) plays TTS back as a *cancellable* task. When
the supplied :class:`asyncio.Event` is set, the playback task is cancelled
promptly, no further audio is emitted, and the pipeline re-enters ``LISTENING``.
A streaming TTS (one exposing an async-generator ``stream``) is consumed chunk by
chunk so cancellation can land mid-stream; a plain :class:`TTSProvider` is played
as a single awaitable.

No heavy voice library and no LLM SDK is imported here — the pipeline depends only
on the typed boundaries (wake/capture/VAD/STT/TTS protocols and the orchestrator).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.providers.stt import STTProvider
from friday.providers.tts import TTSProvider, VoiceConfig
from friday.voice.capture import AudioCapture
from friday.voice.vad import VAD
from friday.voice.wake_word import WakeWordDetector

# A listener notified on every Mode transition the pipeline emits.
ModeListener = Callable[[Mode], None]


@runtime_checkable
class StreamingTTS(Protocol):
    """A :class:`TTSProvider` that can also yield audio in cancellable chunks.

    ``stream`` is the barge-in-friendly path: it yields audio chunks one at a
    time so playback can be cancelled mid-utterance. Any provider implementing
    this protocol is preferred by :meth:`VoicePipeline.speak_with_bargein`;
    providers without it are played as a single :meth:`synthesize` await.
    """

    def stream(self, text: str, voice: VoiceConfig) -> AsyncIterator[bytes]:
        """Yield synthesized audio for ``text`` in cancellable chunks."""
        ...


class VoiceTurn(BaseModel):
    """The result of one :meth:`VoicePipeline.run_once` turn.

    Attributes:
        transcript: The STT transcript of the captured utterance.
        response_text: The orchestrator's synthesized reply.
        mode: The final core :class:`~friday.core.state.Mode` for the turn.
        audio: The synthesized reply as audio bytes (never empty on success).
    """

    transcript: str
    response_text: str
    mode: Mode
    audio: bytes


class VoicePipeline:
    """Drives the spoken-turn loop over the voice boundaries.

    Args:
        wake: Wake-word detector evaluated frame by frame over the capture stream.
        capture: The audio source yielding raw PCM frames.
        vad: Voice-activity detector used to find the end of the utterance.
        stt: Speech-to-text provider.
        orchestrator: The core orchestrator producing the reply.
        tts: Text-to-speech provider (optionally a :class:`StreamingTTS`).
        voice: Voice-selection config passed to the TTS provider.
    """

    def __init__(
        self,
        wake: WakeWordDetector,
        capture: AudioCapture,
        vad: VAD,
        stt: STTProvider,
        orchestrator: Orchestrator,
        tts: TTSProvider,
        voice: VoiceConfig | None = None,
    ) -> None:
        self._wake = wake
        self._capture = capture
        self._vad = vad
        self._stt = stt
        self._orchestrator = orchestrator
        self._tts = tts
        self._voice = voice if voice is not None else VoiceConfig()
        self._mode: Mode = Mode.IDLE
        self._listeners: list[ModeListener] = []

    # -- mode emission ----------------------------------------------------- #
    def on_mode(self, listener: ModeListener) -> None:
        """Register ``listener`` to be called on every mode transition."""
        self._listeners.append(listener)

    @property
    def mode(self) -> Mode:
        """The pipeline's current :class:`~friday.core.state.Mode`."""
        return self._mode

    def _set_mode(self, mode: Mode) -> None:
        """Transition to ``mode`` and notify every registered listener."""
        self._mode = mode
        for listener in self._listeners:
            listener(mode)

    # -- one turn ---------------------------------------------------------- #
    async def run_once(self, session_id: str) -> VoiceTurn | None:
        """Drive one Idle->Listening->Routing turn; return the :class:`VoiceTurn`.

        Awaits a wake detection over ``capture.frames()``; once the wake word
        fires, captures the utterance frames until the VAD reports silence,
        transcribes, runs the orchestrator, and synthesizes the reply. Returns
        ``None`` if the capture stream ends before any wake detection (no turn).
        """
        self._set_mode(Mode.IDLE)

        frames = self._capture.frames()
        woke = await self._await_wake(frames)
        if not woke:
            self._set_mode(Mode.IDLE)
            return None

        self._set_mode(Mode.LISTENING)
        utterance = await self._capture_utterance(frames)

        self._set_mode(Mode.ROUTING)
        transcript = await self._stt.transcribe(utterance, lang=None)

        state = GraphState(session_id=session_id, user_input=transcript.text)
        result = await self._orchestrator.handle(state)
        response_text = result.response or ""

        audio = await self._tts.synthesize(response_text, self._voice)

        return VoiceTurn(
            transcript=transcript.text,
            response_text=response_text,
            mode=result.mode,
            audio=audio,
        )

    async def _await_wake(self, frames: AsyncIterator[bytes]) -> bool:
        """Consume frames until the wake word fires; ``False`` if the stream ends."""
        async for frame in frames:
            if self._wake.detect(frame).detected:
                return True
        return False

    async def _capture_utterance(self, frames: AsyncIterator[bytes]) -> bytes:
        """Accumulate frames until the VAD reports silence (or the stream ends).

        The VAD is polled per frame; once it reports a non-speech frame the
        utterance is considered complete. Frames consumed before the first
        silence are concatenated into the raw audio payload handed to STT.
        """
        buffer = bytearray()
        async for frame in frames:
            if not self._vad.is_speech(frame):
                break
            buffer.extend(frame)
        return bytes(buffer)

    # -- barge-in ---------------------------------------------------------- #
    async def speak_with_bargein(self, text: str, bargein: asyncio.Event) -> None:
        """Play ``text`` as TTS, cancelling promptly when ``bargein`` is set.

        Playback runs as a cancellable task. A second task waits on the
        ``bargein`` event; whichever finishes first wins. If barge-in fires, the
        playback task is cancelled (so no further audio is emitted) and the
        pipeline re-enters ``LISTENING``. Otherwise playback completes normally
        and the pipeline returns to ``IDLE``.
        """
        play_task = asyncio.ensure_future(self._play(text))
        bargein_task = asyncio.ensure_future(bargein.wait())
        try:
            done, pending = await asyncio.wait(
                {play_task, bargein_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bargein_task in done and not play_task.done():
                # Barge-in won the race: stop playback immediately.
                play_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await play_task
                self._set_mode(Mode.LISTENING)
                return
            # Playback finished first (or exactly at the signal): surface any
            # playback error and tear the barge-in waiter down.
            await play_task
            self._set_mode(Mode.IDLE)
        finally:
            if not bargein_task.done():
                bargein_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await bargein_task

    async def _play(self, text: str) -> None:
        """Render ``text`` to audio, streaming chunk by chunk when supported.

        A :class:`StreamingTTS` is consumed via its async-generator ``stream`` so
        a cancellation can land between chunks; a plain provider is played as a
        single :meth:`~friday.providers.tts.TTSProvider.synthesize` await.
        """
        tts = self._tts
        if isinstance(tts, StreamingTTS):
            async for _chunk in tts.stream(text, self._voice):
                # Each chunk is "played"; a cancellation here stops emission.
                pass
            return
        await tts.synthesize(text, self._voice)
