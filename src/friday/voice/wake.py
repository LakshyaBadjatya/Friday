# © Lakshya Badjatya — Author
"""Wake-word + summon command parsing, and per-operator voice styles.

The voice layer turns a transcribed phrase into a structured command:

* **"Hey FRIDAY"** -> a ``wake`` command — boot the cockpit and have FRIDAY greet.
* **"FRIDAY, summon VISION"** / **"spawn GECKO"** -> a ``summon`` command naming a
  roster operator — route the turn under that persona, which answers in its own
  voice.

This module is the **pure core**: parsing is deterministic regex work over a
transcript plus the known operator names, and :data:`OPERATOR_VOICES` is plain
data the HUD reads to give each operator a distinct browser ``speechSynthesis``
voice. It imports no audio/ML library — mic capture + the wake-word model live
behind the openwakeword seam, and the browser does the speaking — so the command
grammar can be unit-tested exhaustively and offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

#: A parsed command is either a bare wake or a summon naming an operator.
WakeKind = Literal["wake", "summon"]


class WakeCommand(BaseModel):
    """A recognized wake/summon command.

    ``kind="wake"`` is a bare "Hey FRIDAY"; ``kind="summon"`` carries the resolved
    roster ``operator`` code-name (e.g. ``"VISION"``).
    """

    kind: WakeKind
    operator: str | None = None


# "hey/hi friday" (tolerant of a dropped 'd' mis-hearing) anywhere in the phrase.
_WAKE_RE = re.compile(r"\b(?:hey|hi|ok|okay)\s*,?\s+frid?ay\b", re.IGNORECASE)
# "friday summon/spawn/call/get/bring up <name>" -> summon that operator.
_SUMMON_RE = re.compile(
    r"\bfrid?ay\s*,?\s+(?:summon|spawn|call|get|bring\s+up|wake)\s+([a-z]+)\b",
    re.IGNORECASE,
)


def parse_wake_command(
    transcript: str, operators: Iterable[str]
) -> WakeCommand | None:
    """Parse a transcript into a :class:`WakeCommand`, or ``None`` if it isn't one.

    Summon is checked first (it is the more specific phrase): a "FRIDAY summon
    <name>" whose ``<name>`` resolves (case-insensitively) to one of ``operators``
    yields a summon; a named-but-unknown operator yields ``None`` (we don't guess
    at an operator we don't have). Otherwise a "Hey FRIDAY" anywhere yields a wake.
    """
    text = transcript.strip()
    if not text:
        return None
    known = {op.upper(): op for op in operators}
    summon = _SUMMON_RE.search(text)
    if summon is not None:
        name = summon.group(1).upper()
        if name in known:
            return WakeCommand(kind="summon", operator=known[name])
        return None
    if _WAKE_RE.search(text):
        return WakeCommand(kind="wake")
    return None


class VoiceStyle(BaseModel):
    """A browser ``speechSynthesis`` style giving one operator a distinct voice.

    ``pitch`` (0..2) and ``rate`` (0.1..10) shape the timbre/speed; ``hint`` is a
    case-insensitive substring the HUD prefers when picking a system voice (e.g.
    ``"female"`` / ``"male"`` / a specific voice name).
    """

    pitch: float = Field(default=1.0, ge=0.0, le=2.0)
    rate: float = Field(default=1.0, ge=0.1, le=10.0)
    hint: str = ""


#: A distinct voice per roster operator (the prime FRIDAY + eight specialists),
#: so each speaks in its own timbre. The HUD reads this to configure the browser
#: voice for whichever operator is replying.
OPERATOR_VOICES: dict[str, VoiceStyle] = {
    "FRIDAY": VoiceStyle(pitch=1.15, rate=1.02, hint="female"),
    "EDITH": VoiceStyle(pitch=0.9, rate=1.08, hint="female"),
    "ORACLE": VoiceStyle(pitch=1.0, rate=0.92, hint="female"),
    "GECKO": VoiceStyle(pitch=0.7, rate=1.0, hint="male"),
    "KAREN": VoiceStyle(pitch=1.3, rate=1.12, hint="female"),
    "VERONICA": VoiceStyle(pitch=1.08, rate=1.05, hint="female"),
    "JOCASTA": VoiceStyle(pitch=0.95, rate=0.88, hint="female"),
    "VISION": VoiceStyle(pitch=0.85, rate=0.98, hint="male"),
    "FORGE": VoiceStyle(pitch=0.62, rate=1.0, hint="male"),
}


def voice_for(operator: str) -> VoiceStyle:
    """Return the :class:`VoiceStyle` for ``operator`` (a neutral default if unknown)."""
    return OPERATOR_VOICES.get(operator.upper(), VoiceStyle())
