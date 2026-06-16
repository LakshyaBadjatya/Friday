# © Lakshya Badjatya — Author
"""Per-turn cost/latency budgeter — a pure, offline spend governor.

A turn against a free/free-tier model is still bounded: the orchestrator wants to
cap how many tokens (and, where priced, how many dollars) a single conversation
turn may consume, and to *downshift* to a cheaper/smaller model once a turn is
running hot. This module owns that arithmetic. :class:`TurnBudget` is a single
turn's running tally against its caps; :class:`Budgeter` keeps one such tally per
session and answers the two questions the orchestrator actually asks — "are we
over budget?" and "should we downshift the active model now?".

Everything here is pure and deterministic: it imports no LLM SDK, reads no
:func:`~friday.config.get_settings` (the caps arrive as constructor params via
``app.py``'s ``_build_budgeter``), opens no socket, and touches no wall clock.
The orchestrator records each completion's usage with :meth:`Budgeter.record`,
then consults :meth:`Budgeter.should_downshift` to decide whether to swap the
:class:`~friday.models.gateway.ModelGateway`'s active model down a tier.
"""

from __future__ import annotations

from pydantic import BaseModel


class TurnBudget(BaseModel):
    """One turn's running spend against its token (and optional dollar) caps.

    ``max_tokens`` is the hard token ceiling for the turn; ``max_usd`` is an
    optional dollar ceiling (``None`` = unpriced, e.g. a free model). ``spent_*``
    accumulate as the turn runs via :meth:`record`. The budget is considered
    exhausted (:meth:`over_budget`) once spend *reaches* a cap — ``>=`` not ``>``
    — so a turn that lands exactly on the ceiling is treated as full, never
    allowed one more over-the-line call.
    """

    max_tokens: int
    max_usd: float | None = None
    spent_tokens: int = 0
    spent_usd: float = 0.0

    def record(self, tokens: int, usd: float = 0.0) -> None:
        """Add one completion's ``tokens`` (and optional ``usd``) to the tally."""
        self.spent_tokens += tokens
        self.spent_usd += usd

    def remaining_tokens(self) -> int:
        """Tokens left before the cap, clamped at zero (never negative)."""
        return max(0, self.max_tokens - self.spent_tokens)

    def over_budget(self) -> bool:
        """Whether the turn has reached either cap.

        ``True`` once ``spent_tokens >= max_tokens``, or — when a dollar cap is
        set — once ``spent_usd >= max_usd``. The boundary is inclusive (``>=``):
        spending exactly the cap counts as over, so the next call is withheld.
        """
        if self.spent_tokens >= self.max_tokens:
            return True
        return self.max_usd is not None and self.spent_usd >= self.max_usd


class Budgeter:
    """Per-session turn budgeter answering "over budget?" and "downshift now?".

    Holds one :class:`TurnBudget` per ``session_id`` so concurrent sessions never
    share a tally. ``max_tokens`` and the optional ``max_usd`` are the per-turn
    caps applied to every session's budget; ``downshift_at`` is the fraction of
    the token cap (``0.8`` = 80 %) at or beyond which :meth:`should_downshift`
    trips, letting the orchestrator drop the gateway's active model to a cheaper
    tier *before* the turn is fully exhausted.

    Deterministic and clock-free: it tracks only counts and dollars, so no
    injected clock is needed. The caps are injected (not read from settings) so
    the type stays offline-first and trivially testable.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        max_usd: float | None = None,
        downshift_at: float = 0.8,
    ) -> None:
        self._max_tokens = max_tokens
        self._max_usd = max_usd
        self._downshift_at = downshift_at
        self._budgets: dict[str, TurnBudget] = {}

    def start_turn(self, session_id: str) -> TurnBudget:
        """Reset ``session_id``'s budget to a fresh turn and return it.

        Replaces any prior tally for the session with a new zeroed
        :class:`TurnBudget` carrying the configured caps, so per-turn spend never
        leaks across turns. Returns the new budget for the caller to inspect.
        """
        budget = TurnBudget(max_tokens=self._max_tokens, max_usd=self._max_usd)
        self._budgets[session_id] = budget
        return budget

    def _budget(self, session_id: str) -> TurnBudget:
        """Return the session's budget, lazily starting a turn if none exists.

        A :meth:`record` or query before an explicit :meth:`start_turn` is treated
        as the start of the session's first turn rather than an error, so the
        orchestrator never has to special-case the very first call.
        """
        budget = self._budgets.get(session_id)
        if budget is None:
            budget = self.start_turn(session_id)
        return budget

    def record(self, session_id: str, tokens: int, usd: float = 0.0) -> None:
        """Record one completion's ``tokens`` (and optional ``usd``) for the session."""
        self._budget(session_id).record(tokens, usd)

    def should_downshift(self, session_id: str) -> bool:
        """Whether the session's turn has crossed the downshift threshold.

        ``True`` once the session has spent at least ``downshift_at`` of the token
        cap (``spent_tokens >= downshift_at * max_tokens``) or — when a dollar cap
        is set — has reached it (:meth:`TurnBudget.over_budget` on the dollar
        side). The orchestrator uses this to swap the gateway's active model down
        a tier; it is intentionally a softer trip than :meth:`TurnBudget.over_budget`.
        """
        budget = self._budget(session_id)
        if budget.spent_tokens >= self._downshift_at * self._max_tokens:
            return True
        return self._max_usd is not None and budget.spent_usd >= self._max_usd

    def remaining(self, session_id: str) -> int:
        """Tokens left in the session's current turn (clamped at zero)."""
        return self._budget(session_id).remaining_tokens()
