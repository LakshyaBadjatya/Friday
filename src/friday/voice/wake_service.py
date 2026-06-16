# © Lakshya Badjatya — Author
"""Wake service: turn a transcript into a wake/summon event with a spoken greeting.

A transcript arrives (from the browser's speech recognition over ``/ws/wake``, or
later from a server-side STT runner over the wake-word engine). This service parses
it with :func:`friday.voice.wake.parse_wake_command` and, on a match, builds the
:class:`WakeEvent` the HUD acts on: reveal the cockpit, and have the right operator
speak the greeting **in its own voice**.

* "Hey FRIDAY" -> ``wake`` — FRIDAY says "I'm up, &lt;owner&gt;."
* "FRIDAY summon VISION" -> ``summon`` — VISION says "VISION here, &lt;owner&gt;."

Pure and deterministic — no audio/ML import — so the whole wake/summon decision is
unit-testable; the mic capture + WebSocket transport live in the app layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from friday.voice.wake import WakeKind, parse_wake_command

#: The operator that greets on a bare wake (the prime).
PRIME = "FRIDAY"


class WakeEvent(BaseModel):
    """What the HUD should do for a recognized wake/summon.

    Attributes:
        type: ``"wake"`` or ``"summon"``.
        operator: The operator who speaks (the prime for a bare wake; the summoned
            operator otherwise) — the HUD picks that operator's voice.
        greeting: The line the operator speaks aloud.
    """

    type: WakeKind
    operator: str
    greeting: str


class WakeService:
    """Parses transcripts into :class:`WakeEvent`s, addressing the owner by name.

    Args:
        operators: The roster code-names a summon may name.
        owner_address: How operators address the owner in greetings (e.g. "Boss").
    """

    def __init__(self, operators: Iterable[str], *, owner_address: str = "Boss") -> None:
        self._operators = list(operators)
        self._owner = owner_address

    def handle_transcript(self, transcript: str) -> WakeEvent | None:
        """Return the :class:`WakeEvent` for ``transcript``, or ``None`` if it isn't one."""
        command = parse_wake_command(transcript, self._operators)
        if command is None:
            return None
        if command.kind == "summon" and command.operator:
            return WakeEvent(
                type="summon",
                operator=command.operator,
                greeting=f"{command.operator} here, {self._owner}.",
            )
        return WakeEvent(
            type="wake", operator=PRIME, greeting=f"I'm up, {self._owner}."
        )
