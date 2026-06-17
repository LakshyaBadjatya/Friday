# © Lakshya Badjatya — Author
"""Turn replay — a bounded transcript of recent turns for the dashboard.

Where the :class:`~friday.observability.tracing.Tracer` keeps *timings* (route /
dispatch / synth spans) and :class:`~friday.observability.metrics.Metrics` keeps
*counts*, the recorder keeps the *content*: the exact user input and assistant
reply (plus session + mode) of each turn, in a ring buffer. ``GET /admin/turns``
lists them and ``GET /admin/turns/{id}`` fetches one, so the local dashboard can
show — and a human can replay/inspect — what actually went in and came out.

Each turn gets a monotonic integer id from an in-process counter (no clock, no
randomness — both would break determinism), so ids are stable and orderable
within a process. Pure and offline: in-memory only, no I/O, no LLM SDK.
"""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel


class TurnRecord(BaseModel):
    """One captured turn: its id, session, mode, and the input/output text.

    ``mode`` is the turn's final :class:`~friday.core.modes.Mode` rendered to a
    string (``None`` if the turn never reached routing). ``response`` is ``None``
    only if the turn produced no reply at all (it carries the honest error
    message on a failed turn, since the recorder runs after the turn settles).
    """

    id: int
    session_id: str
    mode: str | None
    user_input: str
    response: str | None


class TurnRecorder:
    """Bounded, in-process recorder of recent turn transcripts."""

    def __init__(self, capacity: int = 256) -> None:
        self._turns: deque[TurnRecord] = deque(maxlen=capacity)
        self._next_id = 1

    def record(
        self,
        *,
        session_id: str,
        user_input: str,
        response: str | None,
        mode: str | None,
    ) -> TurnRecord:
        """Append a turn transcript, assigning the next monotonic id; return it."""
        record = TurnRecord(
            id=self._next_id,
            session_id=session_id,
            mode=mode,
            user_input=user_input,
            response=response,
        )
        self._next_id += 1
        self._turns.append(record)
        return record

    def recent(self, limit: int = 50) -> list[TurnRecord]:
        """Return up to ``limit`` most-recent turns, oldest-first (newest last)."""
        turns = list(self._turns)
        if limit >= 0:
            turns = turns[-limit:] if limit else []
        return turns

    def get(self, turn_id: int) -> TurnRecord | None:
        """Return the turn with ``turn_id``, or ``None`` if it has aged out / never was."""
        for record in self._turns:
            if record.id == turn_id:
                return record
        return None
