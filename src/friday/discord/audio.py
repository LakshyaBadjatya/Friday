"""Turning speech into text and text back into speech, for free.

Both halves were verified against the live APIs before any of this was written,
because the whole free-tier plan hinged on them:

* **Hearing** — decoded PCM is wrapped in a WAV header (44 bytes of struct
  packing, no library) and sent to Gemini, which transcribes audio natively on
  the key the assistant already uses. No Whisper, so no gigabyte of model weights
  and no CPU budget the free tier does not have.
* **Speaking** — ``edge-tts`` needs no API key and no account at all, which is
  rare enough to be worth stating. It returns MP3; ffmpeg turns that into the raw
  PCM the Opus encoder wants.

ffmpeg is used purely as a format converter, never for network access, and it is
fed through a pipe rather than a shell string — the words she speaks are
model-generated and must never be able to reach a shell.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import urllib.request
from typing import Any

import anyio

from friday.discord import lang, opus, prosody
from friday.logging import get_logger

logger = get_logger("friday.discord.audio")

#: Voice long enough to be worth transcribing. Below this it is a cough, a chair,
#: or the first syllable of someone changing their mind.
MIN_SPEECH_BYTES = opus.FRAME_BYTES * 25  # ~0.5s
#: Ceiling on one utterance, so a hot mic cannot grow the buffer without bound or
#: hand the transcriber a ten-minute file.
MAX_SPEECH_BYTES = opus.FRAME_BYTES * 50 * 30  # ~30s

_STT_PROMPT = (
    "Transcribe this audio exactly. Reply with the two-letter language code, a "
    "pipe, then the words: for example 'en|hello there' or 'pl|dzien dobry'. "
    "Nothing else — no commentary, no speaker labels, no timestamps. If there is "
    "no intelligible speech, reply with nothing at all."
)


def wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV header.

    Gemini needs a container to know the sample rate and channel count; raw PCM
    carries neither. The header is 44 bytes of struct packing, so this avoids
    both a dependency and an ffmpeg round trip for something entirely mechanical.
    """
    channels, rate, bits = opus.CHANNELS, opus.SAMPLE_RATE, 16
    byte_rate = rate * channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, rate, byte_rate,
            channels * bits // 8, bits,
        )
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


async def transcribe(settings: Any, pcm: bytes) -> tuple[str, str] | None:
    """Speech to ``(language_code, text)``, or ``None`` when nothing was said.

    The language comes back with the words rather than from a separate detection
    pass: nobody announces switching to Polish mid-call, they just do it, and the
    transcriber already knows which language it just heard. ``None`` is a normal
    outcome — most of what a voice channel carries is breathing, keyboards and
    background television.
    """
    secret = getattr(settings, "gemini_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if not key or len(pcm) < MIN_SPEECH_BYTES:
        return None
    audio = wav(pcm[:MAX_SPEECH_BYTES])
    model = str(getattr(settings, "stt_model", "") or "gemini-2.5-flash")
    raw = await anyio.to_thread.run_sync(_gemini_stt, key, model, audio)
    if not raw:
        return None
    code, _, said = raw.partition("|")
    code = code.strip().lower()[:2]
    said = said.strip()
    # No pipe means the model ignored the format; the words are still good, so
    # take them and assume English rather than throwing the turn away.
    if not said:
        return ("en", raw.strip()) if raw.strip() else None
    return (code or "en"), said


def _gemini_stt(key: str, model: str, audio: bytes) -> str | None:
    """One transcription call. Blocking; callers use a worker thread."""
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {"text": _STT_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio).decode(),
                            }
                        },
                    ]
                }
            ]
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:  # noqa: BLE001 - a failed transcription is silence, not a crash
        logger.warning("voice: transcription failed")
        return None
    return (text or "").strip() or None


async def speak(text: str, voice: str | None = None) -> list[bytes]:
    """Text to a list of Opus frames ready for the voice socket.

    Returns ``[]`` rather than raising when anything in the chain is missing, so
    a broken speech path costs her voice and not the process.
    """
    if not text.strip():
        return []
    try:
        # Hesitations go in here rather than being asked of the model: the model
        # writes what she means, and how she says it is a property of the voice.
        # Keeping them apart means the text reply and the spoken one stay the
        # same sentence, and the fillers never end up in the transcript.
        mp3 = await _tts(prosody.humanize(text), voice or lang.voice_for("en"))
        if not mp3:
            return []
        pcm = await _to_pcm(mp3)
        if not pcm:
            return []
        return await anyio.to_thread.run_sync(_encode_all, pcm)
    except Exception:  # noqa: BLE001 - never let the voice path kill the caller
        logger.exception("voice: speech synthesis failed")
        return []


async def _tts(text: str, voice: str) -> bytes:
    """Synthesise speech with edge-tts (no key, no account)."""
    import edge_tts  # noqa: PLC0415

    rate, pitch = prosody.jitter()
    out = bytearray()
    async for chunk in edge_tts.Communicate(
        text, voice, rate=rate, pitch=pitch
    ).stream():
        if chunk["type"] == "audio":
            out += chunk["data"]
    return bytes(out)


async def _to_pcm(mp3: bytes) -> bytes:
    """Decode MP3 to the exact PCM layout Opus wants, via ffmpeg on a pipe.

    Piped rather than passed as a shell string: these words come from a model,
    and model output must never be able to reach a shell.
    """
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-ac", str(opus.CHANNELS), "-ar", str(opus.SAMPLE_RATE),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        pcm, err = await asyncio.wait_for(process.communicate(mp3), timeout=30)
    except TimeoutError:
        process.kill()
        logger.warning("voice: ffmpeg timed out")
        return b""
    if process.returncode != 0:
        logger.warning(
            "voice: ffmpeg failed: %s", err[:200].decode("utf-8", "replace")
        )
        return b""
    return pcm


def _encode_all(pcm: bytes) -> list[bytes]:
    """Chop PCM into 20 ms frames and encode each one.

    A trailing partial frame is padded with silence rather than dropped —
    discarding it clips the last syllable, which is exactly where the meaning
    usually is.
    """
    encoder = opus.Encoder()
    try:
        frames = []
        for offset in range(0, len(pcm), opus.FRAME_BYTES):
            chunk = pcm[offset : offset + opus.FRAME_BYTES]
            if len(chunk) < opus.FRAME_BYTES:
                chunk = chunk + b"\x00" * (opus.FRAME_BYTES - len(chunk))
            frames.append(encoder.encode(chunk))
        return frames
    finally:
        encoder.close()
