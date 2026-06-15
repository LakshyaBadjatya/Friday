"""Study / productivity (Tier 2): spaced-repetition flashcards + study sessions.

This package owns FRIDAY's study feature — a pure SM-2 spaced-repetition core
(:mod:`friday.study.srs`) plus a local-first, SQLite-backed store
(:mod:`friday.study.store`) for flashcards and logged study sessions. It is
usable through the flagged ``/study`` REST surface and is off by default behind
``FRIDAY_ENABLE_STUDY``.

The public surface is the pure :func:`~friday.study.srs.sm2` scheduler with its
:class:`~friday.study.srs.ReviewState`, the typed
:class:`~friday.study.store.Flashcard` / :class:`~friday.study.store.StudySession`
models, and the :class:`~friday.study.store.SQLiteStudyStore` adapter.
"""

from __future__ import annotations

from friday.study.srs import ReviewState, sm2
from friday.study.store import Flashcard, SQLiteStudyStore, StudySession

__all__ = [
    "Flashcard",
    "ReviewState",
    "SQLiteStudyStore",
    "StudySession",
    "sm2",
]
