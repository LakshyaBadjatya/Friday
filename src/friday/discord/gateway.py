"""The Discord gateway socket: reading messages, and the status line.

Slash commands arrive over HTTP and need nothing standing. Everything else the
owner asked for — replying when her name comes up, chiming in mid-conversation,
the Rich Presence line — needs a live WebSocket to Discord, so this holds one
open for the life of the process.

It runs inside the web service rather than as a second deployment, which is only
viable because the keepalive cron already stops Render idling the container. If
that workflow ever stops, presence and message replies stop with it; slash
commands keep working, since they do not depend on this.

Written against the gateway protocol directly rather than pulling in a Discord
library: all that is needed here is identify, heartbeat, reconnect and one event
type. A library would add a dependency and its own opinions about the event loop
for a fraction of its surface.

**Memory.** Discord turns are recorded under their own session and never touch
the one Siri, Telegram and the HUD share — this is the private room. The reverse
does not hold: the shared history is read *into* Discord for context, so she
knows here what she was told by voice. Reads flow one way; writes never do.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import urllib.request
from typing import Any, cast

import anyio

from friday.discord import banter, vision
from friday.discord.voice import VoiceConnection
from friday.logging import get_logger

logger = get_logger("friday.discord.gateway")

_GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
_API = "https://discord.com/api/v10"

# Opcodes.
_DISPATCH, _HEARTBEAT, _IDENTIFY = 0, 1, 2
_PRESENCE_UPDATE, _RECONNECT = 3, 7
_INVALID_SESSION = 9

#: GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT. MESSAGE_CONTENT is privileged and
#: must be switched on in the Developer Portal — without it every message arrives
#: with an empty ``content`` and she looks deaf while appearing perfectly healthy.
_INTENTS = (1 << 0) | (1 << 9) | (1 << 15)

#: Reconnect backoff.
_RETRY_BASE = 2.0
_RETRY_MAX = 60.0


async def run(app: Any) -> None:
    """Hold the socket open forever, reconnecting through anything.

    Never raises. A dead gateway costs messages and presence, but it must not
    take the web service with it, so every failure is logged and retried.
    """
    attempt = 0
    while True:
        try:
            await _session(app)
            attempt = 0
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:  # noqa: BLE001 - the socket must always come back
            attempt += 1
            delay = min(_RETRY_BASE * (2 ** min(attempt, 5)), _RETRY_MAX)
            logger.warning("discord gateway dropped; retrying in %.0fs", delay)
            await anyio.sleep(delay)


async def _session(app: Any) -> None:
    """One connection: identify, heartbeat, dispatch, until it closes."""
    from websockets.asyncio.client import connect  # noqa: PLC0415

    settings = getattr(app.state, "settings", None)
    secret = getattr(settings, "discord_bot_token", None)
    token = secret.get_secret_value() if secret is not None else ""
    if not token:
        logger.info("discord gateway: no bot token; not connecting")
        await anyio.sleep(3600)
        return

    async with connect(_GATEWAY, max_size=2**22) as socket:
        hello = json.loads(await socket.recv())
        interval = float(hello["d"]["heartbeat_interval"]) / 1000.0

        await socket.send(
            json.dumps({
                "op": _IDENTIFY,
                "d": {
                    "token": token,
                    "intents": _INTENTS,
                    "properties": {
                        "os": "linux", "browser": "friday", "device": "friday",
                    },
                    "presence": _presence(),
                },
            })
        )
        logger.info("discord gateway connected")

        async with anyio.create_task_group() as tg:
            tg.start_soon(_heartbeat, socket, interval)
            tg.start_soon(_rotate_presence, socket)
            async for raw in socket:
                event = json.loads(raw)
                op = event.get("op")
                name = event.get("t")
                if op == _DISPATCH and name == "MESSAGE_CREATE":
                    tg.start_soon(_on_message, app, token, event.get("d") or {})
                elif op == _DISPATCH and name == "VOICE_STATE_UPDATE":
                    _on_voice_state(app, socket, tg, event.get("d") or {})
                elif op == _DISPATCH and name == "VOICE_SERVER_UPDATE":
                    _on_voice_server(app, event.get("d") or {})
                elif op in (_RECONNECT, _INVALID_SESSION):
                    logger.info("discord gateway asked for a reconnect")
                    tg.cancel_scope.cancel()
                    return


async def _heartbeat(socket: Any, interval: float) -> None:
    """Keep the connection alive; Discord closes a socket that stops beating.

    The first beat is jittered as the protocol asks — otherwise every client
    reconnecting after an outage beats in lockstep.
    """
    await anyio.sleep(interval * (secrets.randbelow(100) / 100.0))
    while True:
        with contextlib.suppress(Exception):
            await socket.send(json.dumps({"op": _HEARTBEAT, "d": None}))
        await anyio.sleep(interval)


# --- presence -------------------------------------------------------------- #
#: The status line, rotating. Shows on the bot's public profile, not only in the
#: private server — the owner was told and chose this anyway. His bot, his call.
_PRESENCE_LINES = (
    ("Watching", "p*rnhub premium 💀"),
    ("Playing", "with Boss's emotional stability"),
    ("Listening to", "Boss lie about studying"),
    ("Watching", "JEE mocks go badly"),
    ("Playing", "hard to get"),
    ("Watching", "two people flirt, badly 👀"),
    ("Listening to", "excuses"),
    ("Playing", "therapist, unpaid"),
    ("Watching", "the group chat decay"),
    ("Playing", "God (unemployed)"),
)
#: Activity type ids in Discord's numbering.
_ACTIVITY = {"Playing": 0, "Listening to": 2, "Watching": 3}
#: Fast enough to notice, slow enough not to be spam.
_ROTATE_SECONDS = 900


def _presence() -> dict[str, Any]:
    verb, what = secrets.choice(_PRESENCE_LINES)
    return {
        "since": None,
        "activities": [{"name": what, "type": _ACTIVITY[verb]}],
        "status": "online",
        "afk": False,
    }


async def _rotate_presence(socket: Any) -> None:
    """Change the status line periodically so it stays funny."""
    while True:
        await anyio.sleep(_ROTATE_SECONDS)
        with contextlib.suppress(Exception):
            await socket.send(json.dumps({"op": _PRESENCE_UPDATE, "d": _presence()}))


# --- voice ------------------------------------------------------------------ #
def _voice(app: Any) -> dict[str, Any]:
    """Live voice connections, keyed by guild."""
    existing = getattr(app.state, "_voice_calls", None)
    if not isinstance(existing, dict):
        existing = {}
        app.state._voice_calls = existing  # noqa: SLF001 - app state is a namespace
    return existing


async def join_voice(app: Any, socket: Any, guild: str, channel: str) -> Any:
    """Ask the gateway to move the bot into a channel and prepare the connection.

    The request goes over the *main* socket; the credentials come back as two
    separate events, which is why the connection object exists before the
    handshake rather than after it.
    """
    calls = _voice(app)
    existing = calls.get(guild)
    if existing is not None and existing.channel_id == channel:
        return existing
    if existing is not None:
        existing.close()
    call = VoiceConnection(app, guild, channel)
    calls[guild] = call
    await socket.send(json.dumps({
        "op": 4,
        "d": {"guild_id": guild, "channel_id": channel,
              "self_mute": False, "self_deaf": False},
    }))
    return call


def _on_voice_state(app: Any, socket: Any, tg: Any, data: dict[str, Any]) -> None:
    """Follow the owner into a channel; leave when the channel empties.

    Two different events share this name: the bot's own state (which carries the
    session id the handshake needs) and everybody else's (which is how she knows
    the owner just walked in).
    """
    me = _self_id(app)
    guild = str(data.get("guild_id") or "")
    channel = data.get("channel_id")
    user = str(data.get("user_id") or "")
    if not guild:
        return

    if user == me:
        call = _voice(app).get(guild)
        if call is not None and channel:
            call.credentials(session_id=str(data.get("session_id") or ""))
        return

    settings = getattr(app.state, "settings", None)
    owner = str(getattr(settings, "discord_owner_id", "") or "")
    if owner and user != owner:
        return  # only the owner pulls her into a call

    if channel:
        tg.start_soon(_follow_into_voice, app, socket, guild, str(channel))
    else:
        call = _voice(app).pop(guild, None)
        if call is not None:
            call.close()


async def _follow_into_voice(app: Any, socket: Any, guild: str, channel: str) -> None:
    """Join the channel and start the listen/speak loop."""
    call = await join_voice(app, socket, guild, channel)

    async def _heard(said: str, _user: str) -> None:
        reply = await _compose(app, said, f"voice:{channel}", forced=True)
        if reply:
            await call.say(reply)

    try:
        await call.run(_heard)
    except Exception:  # noqa: BLE001 - a dropped call must not kill the gateway
        logger.exception("voice: call ended badly")
    finally:
        call.close()
        _voice(app).pop(guild, None)


def _on_voice_server(app: Any, data: dict[str, Any]) -> None:
    """The other half of the handshake: the token and the endpoint to dial."""
    call = _voice(app).get(str(data.get("guild_id") or ""))
    if call is not None:
        call.credentials(
            token=str(data.get("token") or ""),
            endpoint=str(data.get("endpoint") or ""),
        )


# --- messages -------------------------------------------------------------- #
async def _on_message(app: Any, token: str, message: dict[str, Any]) -> None:
    """Decide whether this message deserves a reply, and send one if so."""
    if (message.get("author") or {}).get("bot"):
        return  # never answer another bot, and never answer herself
    content = (message.get("content") or "").strip()
    channel = str(message.get("channel_id") or "")
    if not channel:
        return

    # Being @-mentioned or replied to is being spoken to, as plainly as typing
    # her name. A mention arrives as markup (<@id>) rather than the word
    # "friday", so matching text alone made her ignore the most natural way to
    # address a bot; a reply to her own message is the other obvious one.
    me = _self_id(app)
    mentioned = any(
        str(u.get("id")) == me for u in (message.get("mentions") or [])
    )
    replying_to_her = me and str(
        ((message.get("referenced_message") or {}).get("author") or {}).get("id", "")
    ) == me
    if mentioned or replying_to_her:
        content = _strip_mention(content) or content

    # She looks at what gets posted here. The description is folded into the
    # message as context rather than becoming the reply, so she answers as
    # herself having seen it — the vision model never addresses the room.
    pictures = vision.images_in(message)
    seen: str | None = None
    if pictures and (banter.addressed(content) or not content):
        seen = await vision.describe(getattr(app.state, "settings", None), pictures)

    if not content and seen is None:
        # Something was posted that could not be read. Counted towards
        # interjections but never guessed at — inventing a description is how
        # "a document from your finance folder" happened.
        banter.note_message(app.state, channel)
        return
    if seen:
        content = (
            f"{content}\n\n[attached image, described: {seen}]" if content
            else f"[image posted, described: {seen}] — react to this naturally."
        )

    # Tagging a message and saying "friday remember this" keeps the *tagged*
    # message. Discord puts it in referenced_message, which is the only place
    # that text exists — without reading it, the instruction gets stored and the
    # thing worth keeping is thrown away.
    if banter.is_remember_this(content):
        quoted = (message.get("referenced_message") or {}).get("content") or ""
        await _send(token, channel, _keep(app, quoted.strip()))
        return

    # Sometimes an emoji is the whole reply. Reacting reads far more like someone
    # half-watching the chat than another paragraph does.
    emoji = banter.reaction_emoji(content)
    if emoji is not None:
        await _react(token, channel, str(message.get("id") or ""), emoji)
        banter.note_message(app.state, channel)
        return

    try:
        reply = await _compose(
            app, content, channel,
            forced=bool(mentioned or replying_to_her)
            or bool(seen and not banter.addressed(content)),
        )
    except Exception:  # noqa: BLE001 - one bad message must not kill the socket
        logger.exception("discord message handling failed")
        return
    if reply:
        await _send(token, channel, reply)


async def _compose(
    app: Any, content: str, channel: str, *, forced: bool = False
) -> str | None:
    """The reply, or ``None`` to stay quiet.

    ``forced`` covers an image posted with no words: there is no name to match
    on, but a picture dropped into the chat is worth a reaction.
    """
    # Told to stop, or laughed at: fixed answers, no model, no argument.
    social = banter.reaction(content)
    if social is not None:
        return social

    # "friday wanna talk" — she cannot start a call, so she asks them to.
    if banter.addressed(content) and banter.wants_voice(content):
        guild_call = next(iter(_voice(app).values()), None)
        return banter.come_to_vc(already_connected=guild_call is not None)

    # "what are you doing" is small talk with a punchline attached, not an
    # identity question — she answers it in the same register as her status line.
    if banter.addressed(content):
        doing = banter.doing_reply(content)
        if doing is not None:
            return doing

    if not forced and not banter.addressed(content):
        # Not talking to her. Occasionally she has something to say anyway.
        if banter.should_interject(app.state, channel):
            return banter.interjection()
        return None

    # Two bits that are the same turn with a different instruction attached,
    # rather than separate pipelines that would drift from her normal voice.
    if channel.startswith("voice:") or forced and channel.startswith("voice"):
        content = f"{content}\n\n[{banter.VOICE_REPLY_RULES}]"
    if banter.is_settle(content):
        content = f"{content}\n\n[{banter.SETTLE_PROMPT}]"
    elif banter.is_callback(content):
        content = f"{content}\n\n[{banter.CALLBACK_PROMPT}]"

    from friday.api.routes_siri import _MAX_QUERY, _produce  # noqa: PLC0415

    # cast: _produce only ever touches ``.app.state``, so the stand-in satisfies
    # it at runtime; the annotation says Request because every other caller is one.
    _speech, raw, _mode, _action = await _produce(
        cast("Any", _GatewayRequest(app)), content[:_MAX_QUERY], discord_session(app)
    )
    return raw or None


def _self_id(app: Any) -> str:
    """The bot's own user id — its application id, which Discord keeps identical."""
    settings = getattr(app.state, "settings", None)
    return str(getattr(settings, "discord_application_id", "") or "")


#: ``<@123>`` / ``<@!123>`` — the raw form an @-mention takes in message content.
_MENTION = re.compile(r"<@!?\d+>")


def _strip_mention(content: str) -> str:
    """Remove the mention markup so the question reads as plain words."""
    return _MENTION.sub("", content).strip()


def discord_session(app: Any) -> str:
    """The private Discord thread — separate from every other surface."""
    settings = getattr(app.state, "settings", None)
    return str(getattr(settings, "discord_session", "discord") or "discord")


class _GatewayRequest:
    """The slice of ``Request`` that :func:`_produce` actually touches.

    A gateway message is not an HTTP request, but the whole brain hangs off
    ``request.app.state``. Rather than fork the pipeline for Discord — which is
    how two implementations begin drifting apart — this supplies the attributes
    it needs, so Discord runs the exact same code as every other surface.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.query_params: dict[str, str] = {}
        self.headers: dict[str, str] = {}

    async def body(self) -> bytes:
        """No HTTP body exists for a gateway message.

        ``_produce`` reads the body looking for GPS coordinates on the "near me"
        path. Omitting this raised ``AttributeError`` on *every* Discord message
        — she connected, set her status, and silently failed to answer anything.
        Empty bytes is the honest answer: there are no coordinates in a chat
        message, and that branch correctly declines to fire.
        """
        return b""


def _keep(app: Any, quoted: str) -> str:
    """Store a tagged message as a durable fact and say what was kept.

    The confirmation quotes it back: "saved" alone gives no way to notice the
    wrong message was captured, and a fact recalled weeks later is far too late
    to find out.
    """
    if not quoted:
        return "reply to the message you want me to keep, Boss 🧍"
    store = getattr(app.state, "long_term", None)
    if store is None:
        return "my long-term memory isn't wired up, Boss."
    body = quoted if len(quoted) <= 2000 else quoted[:2000].rstrip() + "…"
    try:
        store.add_fact(body, "discord")
    except Exception:  # noqa: BLE001 - never lose the turn over storage
        logger.exception("discord remember-this failed")
        return "couldn't hold onto that one, Boss."
    preview = body if len(body) <= 140 else body[:140].rstrip() + "…"
    return f"locked in 🔒 — \"{preview}\""


async def _react(token: str, channel: str, message_id: str, emoji: str) -> None:
    """Add an emoji reaction to a message."""
    if not message_id:
        return
    from urllib.parse import quote  # noqa: PLC0415

    url = (
        f"{_API}/channels/{channel}/messages/{message_id}"
        f"/reactions/{quote(emoji)}/@me"
    )
    request = urllib.request.Request(  # noqa: S310
        url,
        data=b"",
        method="PUT",
        headers={
            "Authorization": f"Bot {token}",
            "Content-Length": "0",
            "User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)",
        },
    )

    def _put() -> None:
        try:
            with urllib.request.urlopen(request, timeout=10):  # noqa: S310
                return
        except Exception:  # noqa: BLE001 - a missed reaction is not worth a crash
            logger.warning("discord reaction failed")

    await anyio.to_thread.run_sync(_put)


async def _send(token: str, channel: str, content: str) -> None:
    """Post a message to a channel."""
    body = json.dumps({"content": content[:1900]}).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{_API}/channels/{channel}/messages",
        data=body,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)",
        },
    )

    def _post() -> None:
        try:
            with urllib.request.urlopen(request, timeout=15):  # noqa: S310
                return
        except Exception:  # noqa: BLE001 - a failed send is not worth a crash
            logger.warning("discord send failed")

    await anyio.to_thread.run_sync(_post)
