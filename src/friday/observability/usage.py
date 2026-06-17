# © Lakshya Badjatya — Author
"""Process-lifetime usage/cost ledger — the data behind the cost dashboard.

Where :class:`~friday.observability.metrics.Metrics` counts *requests* (plus tool
calls / errors / per-mode), this ledger accounts for *spend*: it accumulates the
token usage — and, where a model is priced, the dollar cost — of every LLM
completion the turn loop records, both in aggregate and broken down per model.
:meth:`UsageLedger.snapshot` is exactly what ``GET /admin/usage`` serves, so the
local-first dashboard can show "completions, prompt/completion/total tokens, $
spent" overall and per model without any external metrics backend.

Deliberately simple and offline: monotonic in-process counters, no clock, no I/O,
no LLM SDK. The default catalog is all free models, so they record ``usd=0.0`` —
the dollar columns stay zero while the token columns still tell the real story.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class _ModelTally:
    """Running totals for a single model id (the per-model breakdown row)."""

    __slots__ = ("completions", "prompt_tokens", "completion_tokens", "usd")

    def __init__(self) -> None:
        self.completions = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usd = 0.0


class UsageLedger:
    """Mutable, in-process token/dollar accounting with a JSON-able snapshot."""

    def __init__(self) -> None:
        self._completions = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usd = 0.0
        self._by_model: defaultdict[str, _ModelTally] = defaultdict(_ModelTally)

    def record(
        self,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        usd: float = 0.0,
    ) -> None:
        """Tally one completion's tokens (and optional dollars) under ``model_id``.

        Bumps both the process-wide totals and ``model_id``'s per-model row, so a
        single call keeps the aggregate and the breakdown consistent.
        """
        self._completions += 1
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._usd += usd
        tally = self._by_model[model_id]
        tally.completions += 1
        tally.prompt_tokens += prompt_tokens
        tally.completion_tokens += completion_tokens
        tally.usd += usd

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-able copy of the totals + per-model breakdown.

        The ``by_model`` mapping and every nested row are freshly built, so a
        caller mutating the snapshot can never corrupt the live tallies.
        ``tokens`` is the prompt+completion sum, surfaced so the dashboard need
        not re-add it; dollar figures are rounded to drop float-accumulation noise.
        """
        by_model = {
            model_id: {
                "completions": t.completions,
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "tokens": t.prompt_tokens + t.completion_tokens,
                "usd": round(t.usd, 6),
            }
            for model_id, t in self._by_model.items()
        }
        return {
            "completions": self._completions,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "tokens": self._prompt_tokens + self._completion_tokens,
            "usd": round(self._usd, 6),
            "by_model": by_model,
        }
