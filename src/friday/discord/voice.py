"""Joining a voice channel, listening, and talking back.

Discord's voice protocol is a second connection entirely: its own WebSocket for
negotiation, a UDP socket for the audio, and RTP packets encrypted with a key
neither of those hands over directly. The sequence is fixed and each step depends
on the one before it:

1. Ask the *main* gateway to move the bot into the channel (opcode 4).
2. Discord answers with two separate events — ``VOICE_STATE_UPDATE`` carries the
   session id, ``VOICE_SERVER_UPDATE`` the token and endpoint. They arrive in
   either order, so both are awaited rather than assumed.
3. Connect to that endpoint, identify, and receive an SSRC and a UDP port.
4. **IP discovery**: send a 74-byte packet and Discord echoes back the public
   address it saw. A NAT-ed container cannot know its own external address, so
   this is not optional.
5. Select the encryption mode and receive the secret key.

Only then does audio flow. Frames are 20 ms, so playback paces itself against a
monotonic deadline rather than sleeping 20 ms per frame — sleep always overshoots
slightly, and the overshoot accumulates into audible drift across a sentence.

**Turn-taking.** She listens per speaker, buffers while someone talks, and acts
when they stop, because a voice channel has no send button. Silence is the only
end-of-turn signal available, which makes the pause length the most consequential
number in this file: too short and she interrupts, too long and she feels slow.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import struct
import time
from typing import Any

import anyio

from friday.discord import audio, lang, opus
from friday.logging import get_logger

logger = get_logger("friday.discord.voice")

# Voice websocket opcodes.
_IDENTIFY, _SELECT_PROTOCOL, _READY, _HEARTBEAT = 0, 1, 2, 3
_SESSION_DESCRIPTION, _SPEAKING, _HELLO = 4, 5, 8

#: Encryption modes, best first. Discord retired the xsalsa20 family in 2024, so
#: these AEAD modes are the live ones; both are available in pynacl.
_PREFERRED_MODES = ("aead_xchacha20_poly1305_rtpsize", "aead_aes256_gcm_rtpsize")

#: How long a speaker must be quiet before their turn counts as finished. Under
#: ~0.7s she talks over someone drawing breath mid-sentence; over ~1.5s the
#: conversation feels dead.
SILENCE_SECONDS = 1.0
#: How often the buffers are checked for a finished turn.
_TICK_SECONDS = 0.25

#: Loudness a frame must reach to count as someone talking rather than room
#: noise. Compared on mean absolute amplitude of 16-bit samples: breathing and
#: fan noise sit far below this, speech sits well above it.
_INTERRUPT_FLOOR = 500


def _loud_enough(pcm: bytes) -> bool:
    """Whether a decoded frame carries actual speech."""
    if len(pcm) < 64:
        return False
    total = 0
    # Every 32nd sample is plenty to judge loudness and keeps this cheap enough
    # to run on every inbound packet.
    step = 64
    count = 0
    for offset in range(0, len(pcm) - 1, step):
        sample = int.from_bytes(pcm[offset : offset + 2], "little", signed=True)
        total += abs(sample)
        count += 1
    return bool(count) and (total / count) > _INTERRUPT_FLOOR


#: The canonical Opus silent frame. Discord's own documentation specifies these
#: three bytes; encoding real silence would produce a larger frame that means
#: the same thing, so the constant is used directly.
_SILENCE = b"\xf8\xff\xfe"


class VoiceConnection:
    """One live connection to one voice channel."""

    def __init__(self, app: Any, guild_id: str, channel_id: str) -> None:
        self.app = app
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.ready = asyncio.Event()
        #: Filled by the main gateway as the two credential events arrive.
        self.session_id: str | None = None
        self.token: str | None = None
        self.endpoint: str | None = None
        self._credentials = asyncio.Event()

        self._ws: Any = None
        self._ssrc = 0
        self._secret: bytes = b""
        self._mode = ""
        self._udp: socket.socket | None = None
        self._remote: tuple[str, int] | None = None
        self._sequence = 0
        self._timestamp = 0
        self._nonce = 0
        self._speaking = False
        #: Per-speaker buffers and decoders — Opus is stateful per stream, so
        #: mixing two people through one decoder produces artefacts.
        self._heard: dict[int, bytearray] = {}
        self._last_heard: dict[int, float] = {}
        self._decoders: dict[int, opus.Decoder] = {}
        #: Who is talking, by SSRC, so a transcript can be attributed.
        self._speakers: dict[int, str] = {}
        #: The language last heard in this call; the reply voice follows it.
        self.language = "en"
        #: Raised while she is talking if anyone else starts. Interrupting is
        #: how people take a turn in conversation — a speaker who has to be
        #: waited out is a recording, not someone you are talking to.
        self._interrupted = asyncio.Event()
        #: Set when Discord rejects the handshake (4006), so the caller knows to
        #: ask for fresh credentials rather than retrying with the same dead ones.
        self._stale = False
        self._closed = False

    def credentials(
        self,
        *,
        session_id: str | None = None,
        token: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Feed in half of the handshake as the gateway delivers it."""
        self.session_id = session_id or self.session_id
        self.token = token or self.token
        self.endpoint = endpoint or self.endpoint
        if self.session_id and self.token and self.endpoint:
            self._credentials.set()

    def stale(self) -> bool:
        """Whether the last attempt died because the session was rejected."""
        return self._stale

    async def run(self, on_speech: Any) -> None:
        """Hold the connection open, calling ``on_speech(text, user_id)`` per turn."""

        try:
            await asyncio.wait_for(self._credentials.wait(), timeout=20)
        except TimeoutError:
            logger.warning("voice: no server update — can the bot join that channel?")
            return

        host = str(self.endpoint).split(":")[0]
        logger.info(
            "voice: identifying on %s (session %s…, token %s…)",
            host, str(self.session_id)[:6], str(self.token)[:6],
        )
        try:
            await self._identify_and_run(host, on_speech)
        except Exception as exc:  # noqa: BLE001 - classify, then let it bubble
            # The close code arrives inside an ExceptionGroup from the task
            # group, so str() on the outer exception says only "1 sub-exception"
            # — checking it directly never matched, and the retry that was
            # supposed to recover a rejected session never fired.
            if _mentions_4006(exc):
                self._stale = True
            raise

    async def _identify_and_run(self, host: str, on_speech: Any) -> None:
        """The handshake and the receive loop."""
        from websockets.asyncio.client import connect  # noqa: PLC0415

        async with connect(f"wss://{host}/?v=8", max_size=2**22) as ws:
            self._ws = ws
            await ws.send(
                json.dumps({
                    "op": _IDENTIFY,
                    "d": {
                        "server_id": self.guild_id,
                        "user_id": _self_id(self.app),
                        "session_id": self.session_id,
                        "token": self.token,
                    },
                })
            )
            async with anyio.create_task_group() as tg:
                async for raw in ws:
                    event = json.loads(raw)
                    op, data = event.get("op"), event.get("d") or {}

                    if op == _HELLO:
                        tg.start_soon(
                            self._heartbeat, ws,
                            float(data["heartbeat_interval"]) / 1000.0,
                        )
                    elif op == _READY:
                        await self._open_udp(ws, data)
                    elif op == _SESSION_DESCRIPTION:
                        self._secret = bytes(data["secret_key"])
                        self._mode = data.get("mode", "")
                        self.ready.set()
                        tg.start_soon(self._listen)
                        tg.start_soon(self._reap_turns, on_speech)
                        logger.info("voice: connected to channel %s", self.channel_id)
                    elif op == _SPEAKING:
                        ssrc = int(data.get("ssrc") or 0)
                        if ssrc:
                            self._speakers[ssrc] = str(data.get("user_id") or "")

    async def _heartbeat(self, ws: Any, interval: float) -> None:
        while not self._closed:
            with contextlib.suppress(Exception):
                await ws.send(
                    json.dumps({
                        "op": _HEARTBEAT,
                        "d": {"t": int(time.time() * 1000), "seq_ack": 0},
                    })
                )
            await anyio.sleep(interval)

    async def _open_udp(self, ws: Any, data: dict[str, Any]) -> None:
        """Open the audio socket and tell Discord where to reach us.

        The IP-discovery exchange exists because a NAT-ed host cannot know the
        address the far end sees. Skipping it sends audio that never arrives,
        with no error reported anywhere.
        """
        self._ssrc = int(data["ssrc"])
        self._remote = (data["ip"], int(data["port"]))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        self._udp = sock

        probe = bytearray(74)
        struct.pack_into(">HHI", probe, 0, 0x1, 70, self._ssrc)
        loop = asyncio.get_running_loop()
        await loop.sock_sendto(sock, bytes(probe), self._remote)
        reply = await asyncio.wait_for(loop.sock_recv(sock, 74), timeout=10)
        public_ip = reply[8:72].split(b"\x00", 1)[0].decode()
        public_port = struct.unpack_from(">H", reply, 72)[0]

        modes = data.get("modes") or []
        mode = next(
            (m for m in _PREFERRED_MODES if m in modes),
            modes[0] if modes else "aead_xchacha20_poly1305_rtpsize",
        )
        await ws.send(
            json.dumps({
                "op": _SELECT_PROTOCOL,
                "d": {
                    "protocol": "udp",
                    "data": {
                        "address": public_ip, "port": public_port, "mode": mode,
                    },
                },
            })
        )

    # -- hearing ------------------------------------------------------------ #
    async def _listen(self) -> None:
        """Decrypt and decode incoming audio, buffered per speaker."""
        loop = asyncio.get_running_loop()
        while not self._closed and self._udp is not None:
            try:
                packet = await loop.sock_recv(self._udp, 4096)
            except Exception:  # noqa: BLE001 - socket closed under us
                return
            if len(packet) < 12 or packet[1] != 0x78:
                continue  # not audio: RTCP, keepalive, noise
            ssrc = struct.unpack_from(">I", packet, 8)[0]
            frame = self._decrypt(packet)
            if not frame:
                continue
            try:
                pcm = self._decoder(ssrc).decode(frame)
            except opus.OpusError:
                continue
            # Somebody talking over her ends her turn immediately. The check is
            # on decoded audio rather than Discord's speaking flag, which fires
            # on mic noise and would cut her off mid-word for a cough.
            if self._speaking and _loud_enough(pcm):
                self._interrupted.set()
            self._heard.setdefault(ssrc, bytearray()).extend(pcm)
            self._last_heard[ssrc] = time.monotonic()

    def _decrypt(self, packet: bytes) -> bytes | None:
        """Recover the Opus frame from an encrypted RTP packet.

        The ``_rtpsize`` modes put a 4-byte nonce at the very end and
        authenticate the 12-byte RTP header, so the payload is what sits between
        them. Getting either boundary wrong produces authentication failures that
        look exactly like a wrong key.
        """
        try:
            from nacl.bindings import (  # noqa: PLC0415
                crypto_aead_aes256gcm_decrypt,
                crypto_aead_xchacha20poly1305_ietf_decrypt,
            )

            header, body, tail = packet[:12], packet[12:-4], packet[-4:]
            nonce = tail + b"\x00" * 20
            if self._mode.startswith("aead_aes256"):
                return bytes(
                    crypto_aead_aes256gcm_decrypt(
                        body, header, nonce[:12], self._secret
                    )
                )
            return bytes(
                crypto_aead_xchacha20poly1305_ietf_decrypt(
                    body, header, nonce[:24], self._secret
                )
            )
        except Exception:  # noqa: BLE001 - a bad packet is dropped, not fatal
            return None

    def _decoder(self, ssrc: int) -> opus.Decoder:
        found = self._decoders.get(ssrc)
        if found is None:
            found = opus.Decoder()
            self._decoders[ssrc] = found
        return found

    async def _reap_turns(self, on_speech: Any) -> None:
        """Notice when someone stops talking and hand over what they said."""
        while not self._closed:
            await anyio.sleep(_TICK_SECONDS)
            now = time.monotonic()
            for ssrc, last in list(self._last_heard.items()):
                if now - last < SILENCE_SECONDS:
                    continue
                pcm = bytes(self._heard.pop(ssrc, b""))
                self._last_heard.pop(ssrc, None)
                if len(pcm) < audio.MIN_SPEECH_BYTES:
                    continue
                settings = getattr(self.app.state, "settings", None)
                heard = await audio.transcribe(settings, pcm)
                if heard:
                    code, said = heard
                    # Follow the speaker. Someone switching language mid-call
                    # never says so, so the transcript is the only signal there
                    # is — and answering Polish in an English voice is worse
                    # than not answering.
                    self.language = code
                    logger.info("voice heard [%s]: %s", code, said[:70])
                    with contextlib.suppress(Exception):
                        await on_speech(said, self._speakers.get(ssrc, ""), code)

    # -- speaking ----------------------------------------------------------- #
    async def say(self, text: str, language: str | None = None) -> None:
        """Speak, pacing frames against a clock rather than sleeping blindly."""
        frames = await audio.speak(text, lang.voice_for(language or self.language))
        if not frames or self._udp is None or self._remote is None:
            return
        self._interrupted.clear()
        await self._set_speaking(True)
        deadline = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            for frame in frames:
                if self._closed:
                    break
                if self._interrupted.is_set():
                    # Cut off mid-sentence on purpose. Finishing the thought
                    # after being interrupted is exactly what makes talking to
                    # a machine feel like waiting for one.
                    logger.info("voice: interrupted, stopping")
                    break
                packet = self._encrypt(frame)
                with contextlib.suppress(Exception):
                    await loop.sock_sendto(self._udp, packet, self._remote)
                self._sequence = (self._sequence + 1) & 0xFFFF
                self._timestamp = (self._timestamp + opus.FRAME_SIZE) & 0xFFFFFFFF
                # Against a running deadline rather than `sleep(0.02)`: sleep
                # always overshoots a little, and the overshoot accumulates into
                # audible drift over a sentence.
                deadline += opus.FRAME_MS / 1000.0
                await anyio.sleep(max(0.0, deadline - time.perf_counter()))
            await self._trailing_silence(loop)
        finally:
            await self._set_speaking(False)

    async def _trailing_silence(self, loop: Any) -> None:
        """Five frames of Opus silence, which Discord's clients expect.

        Without them the receiver keeps interpolating from the last real frame
        and the final word smears into a tail that sounds like a dropout.
        """
        if self._udp is None or self._remote is None:
            return
        for _ in range(5):
            with contextlib.suppress(Exception):
                await loop.sock_sendto(
                    self._udp, self._encrypt(_SILENCE), self._remote
                )
            self._sequence = (self._sequence + 1) & 0xFFFF
            self._timestamp = (self._timestamp + opus.FRAME_SIZE) & 0xFFFFFFFF

    def _encrypt(self, frame: bytes) -> bytes:
        """Wrap one Opus frame in an encrypted RTP packet."""
        from nacl.bindings import (  # noqa: PLC0415
            crypto_aead_aes256gcm_encrypt,
            crypto_aead_xchacha20poly1305_ietf_encrypt,
        )

        header = bytearray(12)
        header[0], header[1] = 0x80, 0x78
        struct.pack_into(
            ">HII", header, 2, self._sequence, self._timestamp, self._ssrc
        )
        # The nonce is a plain counter sent in the clear at the end of the
        # packet. It gets its own counter rather than reusing the RTP sequence
        # number, which wraps at 16 bits — a repeated nonce under one key is the
        # one thing an AEAD mode must never do.
        self._nonce = (self._nonce + 1) & 0xFFFFFFFF
        nonce4 = struct.pack(">I", self._nonce)
        nonce = nonce4 + b"\x00" * 20
        if self._mode.startswith("aead_aes256"):
            body = crypto_aead_aes256gcm_encrypt(
                frame, bytes(header), nonce[:12], self._secret
            )
        else:
            body = crypto_aead_xchacha20poly1305_ietf_encrypt(
                frame, bytes(header), nonce[:24], self._secret
            )
        return bytes(header) + bytes(body) + nonce4

    async def _set_speaking(self, speaking: bool) -> None:
        """Tell Discord audio is coming — this is the green ring in the client."""
        if speaking == self._speaking or self._ws is None:
            return
        self._speaking = speaking
        with contextlib.suppress(Exception):
            await self._ws.send(
                json.dumps({
                    "op": _SPEAKING,
                    "d": {
                        "speaking": 1 if speaking else 0,
                        "delay": 0,
                        "ssrc": self._ssrc,
                    },
                })
            )

    def close(self) -> None:
        """Tear down decoders and the socket."""
        self._closed = True
        for decoder in self._decoders.values():
            with contextlib.suppress(Exception):
                decoder.close()
        self._decoders.clear()
        if self._udp is not None:
            with contextlib.suppress(Exception):
                self._udp.close()
            self._udp = None


def _mentions_4006(exc: BaseException, depth: int = 0) -> bool:
    """Whether a 4006 hides anywhere in an exception and its children."""
    if depth > 4:
        return False
    if "4006" in str(exc):
        return True
    for child in getattr(exc, "exceptions", ()) or ():
        if _mentions_4006(child, depth + 1):
            return True
    cause = exc.__cause__ or exc.__context__
    return bool(cause and _mentions_4006(cause, depth + 1))


def _self_id(app: Any) -> str:
    settings = getattr(app.state, "settings", None)
    return str(getattr(settings, "discord_application_id", "") or "")
