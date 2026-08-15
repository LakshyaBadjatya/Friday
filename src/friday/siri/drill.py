"""Flashcard drilling by voice — the one feature that is better without a screen.

``/study`` is a complete SM-2 spaced-repetition system: decks, a due queue, and
graded reviews that schedule the next sighting. It was reachable only over HTTP,
which is the wrong shape for revision — you cannot drill while walking, cooking,
or lying in the dark, which is exactly when drilling actually happens.

The loop is stateful, and that state is the whole design problem. A card is
spoken, and then the *next* spoken turn is the answer to it — so this module holds
one in-flight card per session and reads the following turn against that card
rather than as a fresh question. :func:`handle` therefore has to run *before* the
general question paths in the route, or "mitochondria" gets answered as a biology
query instead of graded as a reply.

Grading is done by the model against the card's stored back, because a spoken
answer is rarely word-for-word: "powerhouse of the cell" and "it makes ATP" are
both right, and a string compare would fail both. The model returns a 0-5 SM-2
grade, which feeds the existing scheduler untouched.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

#: "quiz me" / "quiz me on biology" / "let's revise chemistry".
_START = re.compile(
    r"\b(?:quiz|test|drill)\s+me(?:\s+on\s+(?P<deck>[\w\s]+?))?\s*$"
    r"|\b(?:let'?s\s+)?(?:revise|study)(?:\s+(?P<deck2>[\w\s]+?))?\s*$",
    re.IGNORECASE,
)
#: Leaving the loop.
_STOP = re.compile(
    r"^\s*(?:stop|done|finish(?:ed)?|that'?s\s+enough|quit|exit|no\s+more)\b",
    re.IGNORECASE,
)
#: "I don't know" — a real answer, graded as a blank rather than a wrong guess.
_BLANK = re.compile(
    r"^\s*(?:i\s+)?(?:don'?t\s+know|no\s+idea|dunno|skip|pass|next)\b", re.IGNORECASE
)

#: SM-2 grade for a confidently-correct spoken answer, and for a blank.
_GRADE_GOOD = 5
_GRADE_BLANK = 0
#: Lowest grade still counted as recall by :mod:`friday.study.srs`.
_GRADE_PASS = 3

_PROMPT = (
    "You are grading one spoken flashcard answer. Reply with ONLY a single digit "
    "0 to 5 on the SM-2 scale: 5 if the answer is correct and complete, 4 if "
    "correct with a small gap, 3 if the gist is right, 2 if partly right, 1 if "
    "mostly wrong, 0 if wrong or empty. Judge meaning, not wording — a correct "
    "answer phrased differently still scores 5. Output the digit and nothing else."
)


def is_drilling(state: Any, session_id: str) -> bool:
    """Whether a card is currently in flight for this session."""
    return _pending(state).get(session_id) is not None


async def handle(
    state: Any, session_id: str, query: str, now: datetime, llm: Any
) -> str | None:
    """Advance the drill: start it, grade an answer, or stop. ``None`` to skip."""
    text = (query or "").strip()
    if not text:
        return None
    store = getattr(state, "study_store", None)
    pending = _pending(state)
    card = pending.get(session_id)

    if card is not None and _STOP.match(text):
        pending.pop(session_id, None)
        return "Stopping there, Boss. Say 'quiz me' whenever you want to pick it up."

    if card is not None:
        return await _grade(state, store, session_id, card, text, now, llm)

    started = _START.search(text)
    if started is None:
        return None
    if store is None:
        return "My flashcards aren't wired up yet, Boss."
    deck = (started.group("deck") or started.group("deck2") or "").strip() or None
    return _ask_next(state, store, session_id, now, deck, prefix="")


async def _grade(
    state: Any,
    store: Any,
    session_id: str,
    card: dict[str, Any],
    answer: str,
    now: datetime,
    llm: Any,
) -> str:
    """Score the spoken answer against the stored back, then serve the next card."""
    if _BLANK.match(answer):
        grade, verdict = _GRADE_BLANK, f"No worries. The answer is {card['back']}."
    else:
        grade = await _model_grade(llm, card, answer)
        verdict = (
            "That's right."
            if grade >= _GRADE_PASS
            else f"Not quite — it's {card['back']}."
        )

    if store is not None:
        try:
            store.review_card(int(card["id"]), grade)
        except Exception:  # noqa: BLE001 - a scheduling failure must not end the drill
            pass
    _pending(state).pop(session_id, None)
    if store is None:
        return verdict
    return _ask_next(state, store, session_id, now, card.get("deck"), prefix=f"{verdict} ")


async def _model_grade(llm: Any, card: dict[str, Any], answer: str) -> int:
    """Ask the model for an SM-2 grade; fall back to a pass on any failure.

    A provider hiccup must not mark a right answer wrong — that would corrupt the
    schedule for a card the user actually knows — so the fallback is generous.
    """
    if llm is None:
        return _GRADE_GOOD
    from friday.providers.llm import Message  # noqa: PLC0415

    try:
        response = await llm.complete(
            [
                Message(role="system", content=_PROMPT),
                Message(
                    role="user",
                    content=(
                        f"Question: {card['front']}\n"
                        f"Correct answer: {card['back']}\n"
                        f"Spoken answer: {answer}"
                    ),
                ),
            ]
        )
    except Exception:  # noqa: BLE001
        return _GRADE_GOOD
    digit = re.search(r"[0-5]", getattr(response, "text", "") or "")
    return int(digit.group()) if digit else _GRADE_GOOD


def _ask_next(
    state: Any,
    store: Any,
    session_id: str,
    now: datetime,
    deck: str | None,
    *,
    prefix: str,
) -> str:
    """Speak the next due card and remember it as the one in flight."""
    try:
        due = list(store.due_cards(now))
    except Exception:  # noqa: BLE001
        return f"{prefix}I couldn't reach your cards, Boss.".strip()

    if deck:
        low = deck.lower()
        due = [c for c in due if low in (getattr(c, "deck", "") or "").lower()]
    if not due:
        where = f" in {deck}" if deck else ""
        return f"{prefix}Nothing due{where}, Boss. You're all caught up.".strip()

    card = due[0]
    _pending(state)[session_id] = {
        "id": card.id,
        "front": card.front,
        "back": card.back,
        "deck": getattr(card, "deck", None),
    }
    left = len(due) - 1
    tail = f" {left} to go." if left else " Last one."
    return f"{prefix}{card.front}{tail}"


def _pending(state: Any) -> dict[str, dict[str, Any]]:
    """The per-session in-flight card map, created on first use."""
    existing = getattr(state, "_drill_cards", None)
    if not isinstance(existing, dict):
        existing = {}
        state._drill_cards = existing  # noqa: SLF001 - app state is a namespace
    return existing
