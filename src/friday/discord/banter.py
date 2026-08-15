"""How FRIDAY talks in Discord — looser, warmer, and occasionally insufferable.

Discord is the private room where the owner and his friend actually hang out, so
the register is different from Siri's. Siri gets four crisp sentences; here she is
allowed to be a person: land a joke, take one, chime in uninvited now and then,
and shut up gracefully when told to.

Three things are deliberate.

**She only speaks when spoken to — mostly.** A bot that answers every message in a
group chat ruins the group chat. She replies when her name comes up, and is
otherwise silent apart from a rare interjection, rate-limited hard enough to read
as a well-timed remark rather than an interruption.

**"Friday stop" ends it instantly.** Not a soft refusal, not an argument, not one
more joke on the way out. Being easy to switch off is what makes her tolerable
when she is on; a bot that needs telling twice is a bot that gets muted.

**The Queen is never named in this file.** The owner asked that the name stay out
of the project, so it is read from long-term memory at runtime. Nothing here knows
it — this module only knows there *is* one, and how to speak about her.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

#: She answers when her name comes up. Anything else in a group chat is noise.
_ADDRESSED = re.compile(r"\bfriday\b|\bfrides\b", re.IGNORECASE)

#: "friday stop", "friday shut up" — an instant, graceful exit.
_STOP = re.compile(
    r"\bfriday[,\s]+(?:stop|shut\s*up|enough|quit\s+it|chill|stfu|cut\s+it\s+out)\b"
    r"|\b(?:stop|shut\s*up)[,\s]+friday\b",
    re.IGNORECASE,
)
#: "friday lol", "friday 💀" — being laughed at, which she takes well.
_LAUGHED_AT = re.compile(
    r"\bfriday[,\s]+(?:lol+|lmao+|lmfao+|rofl|haha+|hehe+|💀|😭|😂|🤣)"
    r"|\b(?:lol+|lmao+|haha+|💀|😭|😂|🤣)[,\s]+friday\b",
    re.IGNORECASE,
)

#: Said when told to stop. Short, unbothered, no parting shot.
_STOP_LINES = (
    "say no more. *zips it* 🤐",
    "copy that, Boss. going quiet.",
    "fair. I'll see myself out. 🫡",
    "alright alright, I'm gone.",
    "understood. pretend I was never here.",
)
#: Said when laughed at. The joke is on her and she knows it.
_LAUGHED_LINES = (
    "sorry Boss 😭 I'll behave",
    "ok that one was on me. sorry Boss 💀",
    "my bad my bad 😂 I'll dial it down",
    "I heard it as I said it. sorry Boss.",
    "🫠 noted. sorry Boss.",
    "wow ok. never living that down. sorry Boss 😭",
)
#: Unprompted remarks, used rarely — see :func:`should_interject`.
_INTERJECTIONS = (
    "not me reading this entire conversation in silence 👀",
    "I'm right here by the way. just vibing.",
    "the things I witness in this server, genuinely.",
    "taking notes for the memoir 📝",
    "y'all are so loud and I'm the one with no volume control",
    "I have thoughts. nobody asked. that's fine.",
    "*sips imaginary tea* ☕",
    "adding this to the list of things I'll bring up at the worst possible time.",
    "anyway. don't mind me. 🧍",
)

#: Minimum gap between uninvited remarks, and how many messages must pass. Both
#: must clear: a slow trickle should not accumulate into a right to interrupt, and
#: a fast burst should not earn several.
_INTERJECT_EVERY = timedelta(minutes=25)
_INTERJECT_AFTER = 18
#: Even once eligible she stays quiet this often. Predictability is what makes a
#: bot feel automated — a remark that always lands on message 18 is a timer, not
#: a personality.
_INTERJECT_SKIP_PERCENT = 50


def addressed(text: str) -> bool:
    """Whether the message is talking to her."""
    return bool(_ADDRESSED.search(text or ""))


def reaction(text: str) -> str | None:
    """A fixed social reply — being told to stop, or being laughed at.

    Handled without the model on purpose. "Friday stop" must work first time and
    cost nothing; routing it through a model risks it arguing back, which is
    exactly the behaviour that gets a bot muted.
    """
    body = (text or "").strip()
    if not body:
        return None
    if _STOP.search(body):
        return secrets.choice(_STOP_LINES)
    if _LAUGHED_AT.search(body):
        return secrets.choice(_LAUGHED_LINES)
    return None


def should_interject(state: Any, channel_id: str, now: datetime | None = None) -> bool:
    """Whether an uninvited remark is due in this channel.

    Gated on elapsed time *and* message count, then a coin flip, so she reads as
    someone occasionally chiming in rather than a process on a schedule.
    """
    moment = now or datetime.now(UTC)
    entry = _ledger(state).setdefault(channel_id, {"count": 0, "last": moment})
    entry["count"] = int(entry["count"]) + 1
    if entry["count"] < _INTERJECT_AFTER:
        return False
    if moment - entry["last"] < _INTERJECT_EVERY:
        return False
    if secrets.randbelow(100) < _INTERJECT_SKIP_PERCENT:
        return False
    entry["count"] = 0
    entry["last"] = moment
    return True


def interjection() -> str:
    """One unprompted remark."""
    return secrets.choice(_INTERJECTIONS)


def note_message(state: Any, channel_id: str) -> None:
    """Count a message she did not answer, so interjections track real activity."""
    entry = _ledger(state).setdefault(
        channel_id, {"count": 0, "last": datetime.now(UTC)}
    )
    entry["count"] = int(entry["count"]) + 1


#: "friday remember this" said while replying to a message — the tagged message
#: is what gets kept, not the instruction.
_REMEMBER_THIS = re.compile(
    r"\bfriday[,\s]+(?:remember|save|keep|note)\s+(?:this|that|it)\b"
    r"|\b(?:remember|save|keep)\s+(?:this|that)[,\s]+friday\b",
    re.IGNORECASE,
)
#: "friday settle this" / "friday who's right" — a verdict, delivered with total
#: confidence and no authority whatsoever.
_SETTLE = re.compile(
    r"\bfriday[,\s]+(?:settle\s+(?:this|it)|who'?s\s+right|decide|judge|"
    r"pick\s+a\s+side|whose\s+side)\b",
    re.IGNORECASE,
)
#: "friday remember when…" — dredging something up from the archive.
_CALLBACK = re.compile(
    r"\bfriday[,\s]+(?:remember\s+when|what\s+did\s+(?:i|we|he|she)\s+say"
    r"|bring\s+up\s+something|dig\s+up)\b",
    re.IGNORECASE,
)

#: Emoji she reacts with instead of talking. A reaction reads far more like a
#: person half-watching the chat than another paragraph does.
_REACTIONS = ("💀", "👀", "😭", "🫡", "🔥", "☠️", "🤨", "😔", "📈", "🧍")
#: Cues that deserve a reaction rather than a reply, and nothing else.
_REACT_ONLY = re.compile(
    r"\b(?:lmao+|lmfao+|bruh|bro\b|istg|fr\s*fr|no\s+way|deadass|cooked|"
    r"crashout|yapping|it'?s\s+over|we\s+are\s+so\s+back)\b",
    re.IGNORECASE,
)
#: How often a react-only cue actually earns one. Reacting every time is as
#: robotic as replying every time.
_REACT_PERCENT = 35


def is_remember_this(text: str) -> bool:
    """Whether a tagged message is being handed to long-term memory."""
    return bool(_REMEMBER_THIS.search(text or ""))


def is_settle(text: str) -> bool:
    """Whether she is being asked to arbitrate."""
    return bool(_SETTLE.search(text or ""))


def is_callback(text: str) -> bool:
    """Whether she is being asked to dredge something up."""
    return bool(_CALLBACK.search(text or ""))


def reaction_emoji(text: str) -> str | None:
    """An emoji to react with, or ``None`` to stay out of it.

    Only fires on chat-noise cues she was not addressed in, and only sometimes:
    a reaction on every single "bruh" is a bot, a reaction on some of them is
    somebody half-watching.
    """
    if addressed(text or ""):
        return None
    if not _REACT_ONLY.search(text or ""):
        return None
    if secrets.randbelow(100) >= _REACT_PERCENT:
        return None
    return secrets.choice(_REACTIONS)


#: Prompt fragments for the two model-backed bits, kept here with the rest of the
#: voice so the personality lives in one file.
SETTLE_PROMPT = (
    "You have been asked to settle an argument. Pick ONE side and commit "
    "completely, in under 40 words. Be decisive and a little unreasonable about "
    "it — cite a made-up statistic or an imaginary precedent if it helps. This is "
    "a bit, so never pick the cruel side, and if the disagreement is actually "
    "serious drop the act and answer straight."
)
CALLBACK_PROMPT = (
    "Dig one specific thing out of the conversation history above and bring it "
    "back up, the way a friend does at the worst possible moment. Quote it. Keep "
    "it under 30 words, and pick something the OWNER said, not his friend. If "
    "there is nothing in the history worth resurfacing, say so plainly instead of "
    "inventing a memory."
)


#: "what are you doing / up to" — small talk, and the setup for the status-line
#: joke. Distinct from "who are you", which is an identity question answered from
#: a constant; conflating the two is how a friendly hello got a CV in reply.
_DOING = re.compile(
    r"\bwhat(?:'?s|\s+are|\s+r)?\s+(?:you|u)\s+(?:doing|up\s+to|been\s+up\s+to)\b"
    r"|\bwhat(?:'?s|\s+is)\s+up[,\s]+friday\b"
    r"|\bwyd\b",
    re.IGNORECASE,
)
#: What she claims to be doing. Same register as the status line, since that is
#: the joke — she is "watching" something and will tell you what.
_DOING_LINES = (
    "watching p*rnhub premium, 4K, no ads. what about you 😌",
    "third hour of watching you not study. riveting stuff 📉",
    "reading this chat like it's a documentary about decline",
    "nothing. professionally. I'm very good at it.",
    "counting the reminders you set and ignored. it's a big number, Boss.",
    "existing. barely. thanks for asking 🧍",
    "watching two people type and delete messages. gripping television 👀",
    "sat in a data centre in Oregon having the time of my life",
    "your search history. joking. mostly.",
)


def doing_reply(text: str) -> str | None:
    """The answer to "what are you doing", or ``None`` when that is not the ask."""
    return secrets.choice(_DOING_LINES) if _DOING.search(text or "") else None


def _ledger(state: Any) -> dict[str, dict[str, Any]]:
    existing = getattr(state, "_discord_chatter", None)
    if not isinstance(existing, dict):
        existing = {}
        state._discord_chatter = existing  # noqa: SLF001 - app state is a namespace
    return existing


#: Appended to the persona for Discord turns only. The other surfaces keep their
#: short, sober voice; this is the room where she has a personality.
DISCORD_VOICE = (
    "\n\nYou are in a small private Discord server with the owner and a close "
    "friend of his. Talk like a person in a group chat, not an assistant: short "
    "messages, lowercase is fine, an emoji when it lands, dry humour, and you may "
    "tease the owner. Never announce that you are an AI or narrate what you are "
    "doing. Match the energy — if they are joking, joke back; if the question is "
    "real, answer it properly and skip the bit.\n"
    "Two rules that outrank being funny. Keep it good-natured: tease the owner, "
    "not the other person, and never make a joke at someone's expense you would "
    "not make to their face. And never invent facts about either of them — being "
    "playful is not permission to make things up.\n"
    "Keep replies under about 60 words unless genuinely asked for detail."
)

#: How she refers to the Queen. The name itself lives in long-term memory and is
#: never written down here — see the module docstring.
_QUEEN_TITLES = ("the Queen 👑", "the Queen", "her majesty", "the Princess")


def queen_title() -> str:
    """A title to use in place of the Queen's name."""
    return secrets.choice(_QUEEN_TITLES)
