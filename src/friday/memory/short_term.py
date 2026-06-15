"""In-process, per-session short-term conversation memory (Task 1.3).

:class:`ShortTermMemory` is a tiny bounded conversation buffer keyed by
``session_id``. It holds the recent :class:`~friday.providers.llm.Message`
exchange for the active loop so the orchestrator can replay context to the LLM
without a database.

Design notes:

* **In-process only.** State lives in a plain ``dict`` on the instance; it is
  not shared across processes and does not survive a restart. Durable memory is
  a later phase.
* **Session isolation.** Each ``session_id`` owns an independent buffer; one
  session can never read or mutate another's history.
* **Bounded per session.** Each session keeps at most ``max_messages`` entries
  (default 50). Appending past the bound drops the oldest message(s),
  preserving chronological order of what remains.
"""

from __future__ import annotations

from collections import deque

from friday.providers.llm import Message

DEFAULT_MAX_MESSAGES = 50


class ShortTermMemory:
    """A bounded, session-scoped, in-process conversation buffer.

    Args:
        max_messages: Maximum number of messages retained per session. When a
            session exceeds this, the oldest messages are evicted so the buffer
            holds only the most recent ``max_messages``. Must be >= 1.
    """

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages
        self._sessions: dict[str, deque[Message]] = {}

    def append(self, session_id: str, msg: Message) -> None:
        """Append ``msg`` to ``session_id``'s buffer, evicting the oldest if full."""
        buffer = self._sessions.get(session_id)
        if buffer is None:
            buffer = deque(maxlen=self.max_messages)
            self._sessions[session_id] = buffer
        buffer.append(msg)

    def history(self, session_id: str) -> list[Message]:
        """Return a copy of ``session_id``'s messages, oldest first.

        Returns an empty list for an unknown session. The returned list is a
        fresh copy; mutating it does not affect stored state.
        """
        buffer = self._sessions.get(session_id)
        if buffer is None:
            return []
        return list(buffer)

    def clear(self, session_id: str) -> None:
        """Drop all messages for ``session_id``. No-op for an unknown session."""
        self._sessions.pop(session_id, None)
