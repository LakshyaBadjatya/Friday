"""Meeting capture (Tier 1): audio -> transcript -> LLM summary + action items.

This package owns FRIDAY's meeting-capture feature — record a meeting, turn its
audio into a transcript via the shared STT provider (real Whisper or
:class:`~friday.providers.stt.FakeSTT`), then ask the live LLM once for a short
summary plus extracted action items. The notes are stored in a local-first,
SQLite-backed store, are listable/deletable, and (when a RAG ingestor is wired)
the transcript is ingested so a meeting is answerable via the existing Knowledge
path.

It reuses existing infrastructure only — the STT + LLM providers, the
Phase-4 SQLite path (``memory_db_path``), and the RAG
:class:`~friday.rag.ingest.DocumentIngestor` — and is off by default behind
``FRIDAY_ENABLE_MEETINGS``. LLM summarization is **non-fatal**: any provider or
parse error degrades to transcript-only notes (summary fallback, no action
items), never raising.

The public surface is the typed :class:`~friday.meetings.capture.MeetingNotes`
model, the :class:`~friday.meetings.capture.MeetingCapture` pipeline, and the
:class:`~friday.meetings.store.SQLiteMeetingStore` adapter.
"""

from __future__ import annotations

from friday.meetings.capture import MeetingCapture, MeetingNotes
from friday.meetings.store import MeetingStore, SQLiteMeetingStore

__all__ = [
    "MeetingCapture",
    "MeetingNotes",
    "MeetingStore",
    "SQLiteMeetingStore",
]
