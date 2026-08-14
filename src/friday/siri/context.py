"""The spoken conversation's context window — what FRIDAY remembers between turns.

``/siri/ask`` answers most turns on the fast path (one persona'd LLM call) rather
than through the orchestrator, so the orchestrator's own
:class:`~friday.memory.short_term.ShortTermMemory` write never happened and every
spoken turn started from zero. "What was the last topic?" had nothing to read.

This module is the seam that fixes that. It reads and writes the *same*
``ShortTermMemory`` instance the orchestrator uses (wired at
``app.state.short_term``), so a voice turn and a ``/chat`` turn on the same
``session_id`` share one history — ask by voice, follow up in the web UI, and the
thread is continuous.

Two rules shape what gets replayed:

* **Bounded turns.** Only the last :data:`DEFAULT_CONTEXT_MESSAGES` messages are
  replayed, so latency and token spend stay flat no matter how long the session
  runs.
* **Bounded characters.** Long turns are dropped oldest-first until the replay
  fits :data:`DEFAULT_CONTEXT_CHARS`, so one pasted wall of text cannot crowd out
  the rest of the window.

:func:`is_recall_question` exists because an empty history is a real state that
must be answered honestly: asked "what were we just talking about?" with nothing
stored, FRIDAY says it does not have the earlier turns rather than inventing a
plausible topic.
"""

from __future__ import annotations

from typing import Any, Protocol

from friday.providers.llm import Message

#: Messages (not turns) replayed into the voice prompt — 12 is ~6 exchanges.
DEFAULT_CONTEXT_MESSAGES = 12
#: Total characters of replayed history; older messages are dropped to fit.
DEFAULT_CONTEXT_CHARS = 4000
#: Per-message cap — one rambling turn is shortened by truncation, not dropped.
_MAX_MESSAGE_CHARS = 1200


class MemoryLike(Protocol):
    """The slice of :class:`ShortTermMemory` this module needs (duck-typed)."""

    def append(self, session_id: str, msg: Message) -> None: ...
    def history(self, session_id: str) -> list[Message]: ...


#: Phrasings that ask FRIDAY what the conversation has been about. Matched only to
#: decide whether an *empty* history deserves an honest "I don't have that" —
#: with history present, the model answers from the replayed window as normal.
_RECALL_TRIGGERS = (
    "last topic",
    "previous topic",
    "what were we talking about",
    "what we were talking about",
    "what did we talk about",
    "what were we discussing",
    "what did we discuss",
    "what was i asking",
    "what did i ask",
    "what did you just say",
    "what did i just say",
    "what was my last question",
    "my last question",
    "recap",
    "remind me what",
    "what have we been talking about",
    "earlier i asked",
)

#: Spoken when a recall question arrives with nothing in the window. Honest about
#: the gap rather than inventing a plausible-sounding earlier topic.
NO_HISTORY_REPLY = (
    "I don't have our earlier turns, Boss — this is the first thing I have in "
    "this session. Ask me again and I'll keep track from here."
)


def is_recall_question(query: str) -> bool:
    """Whether ``query`` asks what the conversation has been about."""
    low = query.lower()
    return any(trigger in low for trigger in _RECALL_TRIGGERS)


def recall(
    memory: Any,
    session_id: str,
    *,
    max_messages: int = DEFAULT_CONTEXT_MESSAGES,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[Message]:
    """Return the bounded, chronological context window for ``session_id``.

    Takes the most recent ``max_messages`` messages, truncates any single
    over-long message, then drops from the *oldest* end until the total fits
    ``max_chars``. Returns ``[]`` for an unknown session, a memory that is not
    wired, or a memory whose ``history`` raises — context is an enhancement and
    must never be the reason a spoken turn fails.
    """
    if memory is None or not hasattr(memory, "history"):
        return []
    try:
        stored = memory.history(session_id)
    except Exception:  # noqa: BLE001 - a broken memory must not break the turn
        return []
    if not stored:
        return []

    window = [_truncate(msg) for msg in list(stored)[-max_messages:]]
    total = sum(len(_content(msg)) for msg in window)
    while window and total > max_chars:
        total -= len(_content(window[0]))
        window.pop(0)
    return window


def remember(memory: Any, session_id: str, user_text: str, reply_text: str) -> None:
    """Record one completed turn (the question and the spoken answer).

    Called on *every* branch that produces a reply — fast path, orchestrator,
    circle, Instagram, TV, distance — so the window reflects the whole
    conversation and not just the turns one particular branch handled. Silently
    does nothing when memory is not wired, and swallows storage errors: failing
    to remember must never turn a good answer into an error.
    """
    if memory is None or not hasattr(memory, "append"):
        return
    user_text = (user_text or "").strip()
    reply_text = (reply_text or "").strip()
    if not user_text and not reply_text:
        return
    try:
        if user_text:
            memory.append(session_id, Message(role="user", content=user_text))
        if reply_text:
            memory.append(session_id, Message(role="assistant", content=reply_text))
    except Exception:  # noqa: BLE001 - never fail a turn over bookkeeping
        return


def _content(msg: Message) -> str:
    """A message's text as a plain string.

    ``Message.content`` is optional — a tool-call turn carries ``None`` — so every
    length calculation here goes through this rather than assuming a string.
    """
    return msg.content or ""


def _truncate(msg: Message) -> Message:
    """Cap one message's length, keeping its head (where the topic lives)."""
    content = _content(msg)
    if len(content) <= _MAX_MESSAGE_CHARS:
        return msg
    return Message(role=msg.role, content=content[:_MAX_MESSAGE_CHARS].rstrip() + "…")
