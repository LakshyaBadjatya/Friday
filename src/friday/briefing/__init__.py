"""Proactive briefing (Tier 1): a digest assembled from local stores.

This package owns FRIDAY's briefing feature — a deterministic digest of
due/overdue/upcoming reminders, recent tool-call activity, and a one-line
metrics summary, plus a time-of-day-aware greeting addressing the owner —
available on demand (the flagged ``GET /briefing``) and as a scheduler action so
a morning/EOD briefing fires on its own. It reuses the reminder store and the
observability stores (audit / metrics); off by default behind
``FRIDAY_ENABLE_BRIEFING``. Optional LLM phrasing is non-fatal: any LLM error
falls back to the structured-only briefing.

The public surface is the typed :class:`~friday.briefing.service.Briefing` /
:class:`~friday.briefing.service.BriefingSection` models and the
:class:`~friday.briefing.service.BriefingService` assembler.
"""

from __future__ import annotations

from friday.briefing.service import Briefing, BriefingSection, BriefingService

__all__ = ["Briefing", "BriefingSection", "BriefingService"]
