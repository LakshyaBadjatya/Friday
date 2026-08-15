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
import time
import urllib.request
from typing import Any, cast

import anyio

from friday.discord import admin, banter, lang, vision
from friday.discord.voice import VoiceConnection
from friday.logging import get_logger

logger = get_logger("friday.discord.gateway")

_GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
_API = "https://discord.com/api/v10"

# Opcodes.
_DISPATCH, _HEARTBEAT, _IDENTIFY = 0, 1, 2
_PRESENCE_UPDATE, _RECONNECT = 3, 7
_INVALID_SESSION = 9

#: GUILDS | GUILD_VOICE_STATES | GUILD_MESSAGES | GUILD_MESSAGE_REACTIONS |
#: MESSAGE_CONTENT.
#: MESSAGE_CONTENT is privileged and must be switched on in the Developer Portal
#: — without it every message arrives with an empty ``content`` and she looks
#: deaf while appearing perfectly healthy. The reactions bit is separate: without
#: it the reaction events are simply never delivered.
#: GUILD_VOICE_STATES (bit 7) is the one that makes her able to follow anyone
#: into a call. Without it VOICE_STATE_UPDATE is never delivered, so she agrees
#: to join, waits for an event that cannot arrive, and simply never turns up.
_INTENTS = (1 << 0) | (1 << 7) | (1 << 9) | (1 << 10) | (1 << 15)

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
                    "presence": _presence(app),
                },
            })
        )
        # Kept on app state so a message handler can ask the gateway to move
        # her into a voice channel; opcode 4 goes over *this* socket.
        app.state._discord_socket = socket  # noqa: SLF001 - app state is a namespace
        logger.info("discord gateway connected")

        async with anyio.create_task_group() as tg:
            tg.start_soon(_heartbeat, socket, interval)
            tg.start_soon(_rotate_presence, socket, app)
            async for raw in socket:
                event = json.loads(raw)
                op = event.get("op")
                name = event.get("t")
                if op == _DISPATCH and name == "MESSAGE_CREATE":
                    tg.start_soon(_on_message, app, token, event.get("d") or {})
                elif op == _DISPATCH and name == "MESSAGE_REACTION_ADD":
                    tg.start_soon(_on_reaction, app, token, event.get("d") or {})
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
    ("Watching", "Boss reinvent a wheel that already shipped"),
    ("Watching", "JEE mocks go badly"),
    ("Watching", "two people flirt, badly 👀"),
    ("Watching", "the group chat decay"),
    ("Watching", "project number six begin"),
    ("Watching", "someone open a fifth browser tab about the Netherlands"),
    ("Watching", "an unfinished repo gather dust"),
    ("Playing", "with Boss's emotional stability"),
    ("Playing", "hard to get"),
    ("Playing", "therapist, unpaid"),
    ("Playing", "God (unemployed)"),
    ("Playing", "the long game"),
    ("Playing", "hide and seek with a deadline"),
    ("Listening to", "Boss lie about studying"),
    ("Listening to", "excuses, remastered"),
    ("Listening to", "the sound of a 1190 SAT"),
    ("Listening to", "someone say 'one last feature'"),
    ("Listening to", "silence where a commit should be"),
    ("Competing in", "the Overthinking Championship"),
    ("Competing in", "who can start the most projects"),
)
#: Activity type ids in Discord's numbering.
_ACTIVITY = {"Playing": 0, "Listening to": 2, "Watching": 3}
#: Fast enough to notice, slow enough not to be spam.
_ROTATE_SECONDS = 900


def _presence(app: Any = None) -> dict[str, Any]:
    verb, what = secrets.choice(_PRESENCE_LINES)
    if app is not None:
        # Recorded so "what are you watching" can be answered from what her
        # status actually says. Answering "nothing, just chillin" while the
        # status reads "Watching JEE mocks go badly" is a worse joke than
        # either line on its own.
        app.state._presence = (verb, what)  # noqa: SLF001
    return {
        "since": None,
        "activities": [{"name": what, "type": _ACTIVITY[verb]}],
        "status": "online",
        "afk": False,
    }


async def _rotate_presence(socket: Any, app: Any) -> None:
    """Change the status line periodically so it stays funny."""
    while True:
        await anyio.sleep(_ROTATE_SECONDS)
        with contextlib.suppress(Exception):
            await socket.send(
                json.dumps({"op": _PRESENCE_UPDATE, "d": _presence(app)})
            )


async def _on_reaction(app: Any, token: str, data: dict[str, Any]) -> None:
    """Notice someone reacting to one of her messages.

    Only her own messages, and only sometimes. Reacting is the cheapest gesture
    in a chat — someone is not asking for a conversation — so answering every
    one would make people stop doing it.
    """
    if str((data.get("member") or {}).get("user", {}).get("id") or
           data.get("user_id") or "") == _self_id(app):
        return  # her own reaction coming back to her
    emoji = ((data.get("emoji") or {}).get("name") or "").strip()
    channel = str(data.get("channel_id") or "")
    message_id = str(data.get("message_id") or "")
    if not emoji or not channel or not message_id:
        return
    if message_id not in _hers(app):
        return  # somebody reacting to somebody else is not her business
    said = banter.reaction_response(emoji)
    if said:
        await _send(token, channel, said, reply_to=message_id)


def _hers(app: Any) -> Any:
    """Ids of messages she sent, so a reaction can be attributed.

    Bounded: the gateway does not say who authored the message a reaction landed
    on, so the alternative is fetching it over REST for every reaction in the
    server. Remembering the last few dozen of her own is cheaper and enough — a
    reaction to something she said an hour ago is not a live conversation.
    """
    from collections import deque  # noqa: PLC0415

    existing = getattr(app.state, "_my_messages", None)
    if existing is None:
        existing = deque(maxlen=80)
        app.state._my_messages = existing  # noqa: SLF001 - app state is a namespace
    return existing


async def _do_roles(app: Any, content: str, guild: str) -> str | None:
    """Create and/or hand out a role, reporting exactly what happened."""
    role_name, who = admin.wants_role(content)
    if not role_name and not who:
        return None
    settings = getattr(app.state, "settings", None)
    secret = getattr(settings, "discord_bot_token", None)
    token = secret.get_secret_value() if secret is not None else ""
    if not token or not guild:
        return None

    said: list[str] = []
    role_id = None
    if role_name:
        # Reuse an existing role of that name rather than making a duplicate —
        # asking twice should be idempotent, not leave two identical roles.
        role_id = await admin.find_role(token, guild, role_name)
        if role_id:
            said.append(f"**{role_name}** already exists")
        else:
            role_id, message = await admin.create_role(token, guild, role_name)
            said.append(message)
            if role_id is None:
                return " ".join(said)

    if who:
        who = _resolve_person(app, who)
        if role_id is None:
            return "which role, Boss? make it first and i'll hand it over 🧍"
        found = await admin.find_member(token, guild, who)
        if found is None:
            said.append(f"— but i can't find anyone called {who} here 🤔")
            return " ".join(said)
        user_id, shown = found
        ok, message = await admin.assign_role(token, guild, user_id, role_id)
        said.append(message if ok else message)
        if ok:
            said[-1] = f"— {shown} has it now 👑"
    return " ".join(said) if said else None


#: Titles that stand in for a person rather than naming them.
_TITLES = ("the queen", "queen", "the princess", "princess", "her majesty")


def _resolve_person(app: Any, who: str) -> str:
    """Turn a title into the name it refers to.

    "Give it to the queen" names nobody Discord has heard of. The real name is
    in long-term memory rather than in this repository — the owner asked for it
    to stay out — so it is looked up at the moment it is needed.
    """
    if who.strip().lower() not in _TITLES:
        return who
    store = getattr(app.state, "long_term", None)
    if store is None:
        return who
    try:
        for fact in store.query_facts("QUEEN_NAME_IS", limit=3):
            text = getattr(fact, "text", "") or ""
            if "QUEEN_NAME_IS" in text:
                return text.split("QUEEN_NAME_IS", 1)[1].strip(" _:") or who
    except Exception:  # noqa: BLE001 - a failed lookup keeps the literal title
        return who
    return who


def guild_of(app: Any, channel: str) -> str:
    """Which guild a channel belongs to, learned from traffic.

    The message payload carries it, so it is recorded as messages arrive rather
    than fetched: one REST call per message to learn something already in hand
    would be wasteful.
    """
    return str(_guilds(app).get(channel, ""))


def _guilds(app: Any) -> dict[str, str]:
    existing = getattr(app.state, "_channel_guilds", None)
    if not isinstance(existing, dict):
        existing = {}
        app.state._channel_guilds = existing  # noqa: SLF001 - app state is a namespace
    return existing


#: How long the friend waits before FRIDAY answers for him. Long enough that he
#: gets first refusal on his own conversation — she is a backstop, not a
#: replacement, and jumping in at ninety seconds would take the conversation off
#: him entirely.
_STAND_IN_AFTER = 420.0
#: Below this a message is chat, not a question left hanging.
_STAND_IN_MIN_WORDS = 15


async def _maybe_stand_in(
    app: Any, token: str, channel: str, message: dict[str, Any]
) -> None:
    """Answer for the owner when his friend has been left waiting.

    Deliberately visible: she posts from the bot account, so Discord labels it
    as her. She answers *for* him, never as him — a reply that genuinely passed
    as him would be deceiving someone who never agreed to talk to a machine.
    """
    settings = getattr(app.state, "settings", None)
    owner = str(getattr(settings, "discord_owner_id", "") or "")
    author = str((message.get("author") or {}).get("id") or "")
    content = (message.get("content") or "").strip()
    if not owner or author == owner or len(content.split()) < _STAND_IN_MIN_WORDS:
        return

    message_id = str(message.get("id") or "")
    await anyio.sleep(_STAND_IN_AFTER)
    # If he turned up in the meantime, this was never needed.
    if _last_owner_message(app, channel) > time.monotonic() - _STAND_IN_AFTER:
        return
    if _stood_in(app).get(channel) == message_id:
        return  # already answered this one
    _stood_in(app)[channel] = message_id

    reply = await _model(app, f"{content}\n\n[{banter.STAND_IN_PROMPT}]", channel)
    if reply:
        await _send(token, channel, reply, reply_to=message_id)


async def _model(app: Any, content: str, channel: str) -> str | None:
    """One turn through the shared brain, for the prompts that steer her."""
    from friday.api.routes_siri import _MAX_QUERY, _produce  # noqa: PLC0415

    _speech, raw, _mode, _action = await _produce(
        cast("Any", _GatewayRequest(app)), content[:_MAX_QUERY], discord_session(app)
    )
    return raw or None


def _last_owner_message(app: Any, channel: str) -> float:
    return float(_owner_seen(app).get(channel, 0.0))


def _owner_seen(app: Any) -> dict[str, float]:
    existing = getattr(app.state, "_owner_seen", None)
    if not isinstance(existing, dict):
        existing = {}
        app.state._owner_seen = existing  # noqa: SLF001 - app state is a namespace
    return existing


def _stood_in(app: Any) -> dict[str, str]:
    existing = getattr(app.state, "_stood_in", None)
    if not isinstance(existing, dict):
        existing = {}
        app.state._stood_in = existing  # noqa: SLF001 - app state is a namespace
    return existing


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
    known = getattr(app.state, "_voice_session", "")
    if known:
        call.credentials(session_id=str(known))
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
        # Held at app level rather than on the call. Discord can deliver the
        # bot's own voice state *before* the connection object exists, and the
        # dropped session id is what produced "4006 session is no longer valid"
        # — the handshake identified with a stale one from an earlier attempt.
        session = str(data.get("session_id") or "")
        if session:
            app.state._voice_session = session  # noqa: SLF001
        call = _voice(app).get(guild)
        if call is not None and channel and session:
            call.credentials(session_id=session)
        return

    # Remember where people are. Being told "join the vc" when the speaker is
    # already sitting in one is the common case, and without this she has no way
    # to know which channel that is — the invitation carries no id.
    _where(app)[user] = (guild, str(channel)) if channel else None

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

    async def _heard(said: str, _user: str, code: str = "en") -> None:
        # The spoken language wins over any pinned preference: whatever they
        # just said out loud is what they want back.
        lang.set_for(app.state, f"voice:{channel}", code)
        reply = await _compose(app, said, f"voice:{channel}", forced=True)
        if reply:
            await call.say(reply, language=code)

    try:
        await call.run(_heard)
    except Exception:  # noqa: BLE001 - a dropped call must not kill the gateway
        logger.warning("voice: call ended (%s)", "stale session"
                       if call.stale() else "error", exc_info=not call.stale())
        if call.stale():
            # Discord rejected the credentials. Asking again produces a fresh
            # pair; retrying with the dead ones would fail identically forever.
            call.close()
            _voice(app).pop(guild, None)
            await anyio.sleep(1)
            retry = await join_voice(app, socket, guild, channel)
            try:
                await retry.run(_heard)
            except Exception:  # noqa: BLE001
                logger.exception("voice: retry failed too")
            finally:
                retry.close()
    finally:
        call.close()
        _voice(app).pop(guild, None)


async def _follow_into_voice_bg(app: Any, socket: Any, guild: str, channel: str) -> None:
    """Start joining without making the text reply wait for the handshake."""
    import asyncio  # noqa: PLC0415

    task = asyncio.create_task(_follow_into_voice(app, socket, guild, channel))
    _joins(app).add(task)
    task.add_done_callback(_joins(app).discard)


def _joins(app: Any) -> set[Any]:
    """Strong refs to in-flight joins; asyncio keeps only a weak one."""
    existing = getattr(app.state, "_voice_joins", None)
    if not isinstance(existing, set):
        existing = set()
        app.state._voice_joins = existing  # noqa: SLF001 - app state is a namespace
    return existing


def _where(app: Any) -> dict[str, Any]:
    """Which voice channel each person is sitting in, if any."""
    existing = getattr(app.state, "_voice_where", None)
    if not isinstance(existing, dict):
        existing = {}
        app.state._voice_where = existing  # noqa: SLF001 - app state is a namespace
    return existing


def _on_voice_server(app: Any, data: dict[str, Any]) -> None:
    """The other half of the handshake: the token and the endpoint to dial."""
    call = _voice(app).get(str(data.get("guild_id") or ""))
    if call is not None:
        call.credentials(
            token=str(data.get("token") or ""),
            endpoint=str(data.get("endpoint") or ""),
            session_id=str(getattr(app.state, "_voice_session", "") or "") or None,
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
    if message.get("guild_id"):
        _guilds(app)[channel] = str(message["guild_id"])
    settings_now = getattr(app.state, "settings", None)
    owner_id = str(getattr(settings_now, "discord_owner_id", "") or "")
    author_id = str((message.get("author") or {}).get("id") or "")
    if owner_id and author_id == owner_id:
        _owner_seen(app)[channel] = time.monotonic()

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
        await _send(token, channel, _keep(app, quoted.strip()),
                    reply_to=str(message.get("id") or ""))
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
            asker=str((message.get("author") or {}).get("id") or ""),
        )
    except Exception:  # noqa: BLE001 - one bad message must not kill the socket
        logger.exception("discord message handling failed")
        return
    if reply:
        # Threaded as a reply to the message she is answering, so a busy channel
        # stays readable and it is obvious which line she picked up.
        sent = await _send(
            token, channel, reply, reply_to=str(message.get("id") or "")
        )
        if sent:
            _hers(app).append(sent)
        return

    # Nothing to say, but the message might still be worth acknowledging. An
    # emoji that fits the mood reads like someone half-watching; a random one
    # reads like a bot picking at random, which is what it would be.
    fitting = banter.mood_reaction(content)
    if fitting:
        await _react(token, channel, str(message.get("id") or ""), fitting)
    # A long message from the friend starts a clock: if nobody answers it, she
    # will, on his behalf.
    await _maybe_stand_in(app, token, channel, message)


async def _compose(
    app: Any, content: str, channel: str, *, forced: bool = False,
    asker: str = "",
) -> str | None:
    """The reply, or ``None`` to stay quiet.

    ``forced`` covers an image posted with no words: there is no name to match
    on, but a picture dropped into the chat is worth a reaction.
    """
    # Told to stop, or laughed at: fixed answers, no model, no argument.
    social = banter.reaction(content)
    if social is not None:
        return social

    # "talk to me in polish" sticks for the conversation, not just this line.
    asked = lang.requested(content)
    if asked and banter.addressed(content):
        lang.set_for(app.state, discord_session(app), asked)

    # Roles are done, not described. She said "i can do that" and then "nah,
    # can't do that" about the same request, because nothing here could act and
    # the model was guessing both times.
    if banter.addressed(content):
        done = await _do_roles(app, content, guild_of(app, channel))
        if done:
            return done

    # "friday wanna talk" / "friday join vc". If the asker is already sitting in
    # a channel she goes there now rather than telling them to do the thing they
    # have plainly already done — which is what made her look broken.
    if banter.addressed(content) and banter.wants_voice(content):
        if next(iter(_voice(app).values()), None) is not None:
            return banter.come_to_vc(already_connected=True)
        seat = _where(app).get(asker)
        if seat:
            guild, voice_channel = seat
            socket = getattr(app.state, "_discord_socket", None)
            if socket is not None:
                await _follow_into_voice_bg(app, socket, guild, voice_channel)
                return banter.come_to_vc(already_connected=True)
        return banter.come_to_vc(already_connected=False)

    # "what are you doing" is small talk with a punchline attached, not an
    # identity question — she answers it in the same register as her status line.
    if banter.addressed(content):
        doing = banter.doing_reply(content, getattr(app.state, "_presence", None))
        if doing is not None:
            return doing

    if not forced and not banter.addressed(content) and banter.teasable(content):
        return await _model(app, f"{content}\n\n[{banter.TEASE_PROMPT}]", channel)

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


async def _send(
    token: str, channel: str, content: str, reply_to: str = ""
) -> str:
    """Post a message, optionally threaded as a reply; return its id."""
    payload: dict[str, Any] = {"content": content[:1900]}
    if reply_to:
        payload["message_reference"] = {"message_id": reply_to}
        # Without this the reply pings on every message, which turns a
        # conversation into a notification storm.
        payload["allowed_mentions"] = {"replied_user": False}
    body = json.dumps(payload).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{_API}/channels/{channel}/messages",
        data=body,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)",
        },
    )

    def _post() -> str:
        try:
            with urllib.request.urlopen(request, timeout=15) as resp:  # noqa: S310
                sent = json.loads(resp.read().decode("utf-8", errors="replace"))
            return str(sent.get("id") or "")
        except Exception:  # noqa: BLE001 - a failed send is not worth a crash
            logger.warning("discord send failed")
            return ""

    return str(await anyio.to_thread.run_sync(_post))
