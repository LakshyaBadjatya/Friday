"""Voice Protocols — named routines of registered tool calls (Tier 1).

A *protocol* is a named, ordered sequence of registered tool invocations that one
trigger fires end-to-end (e.g. "run the Goodnight Protocol"). Protocols run ONLY
tools that are already in the shared :class:`~friday.tools.registry.ToolRegistry`
— there is no arbitrary code execution — and they honor the existing confirm-step:
a side-effecting, non-idempotent step pauses the run for the owner's confirmation.

The slice is local-first (SQLite-persisted), off behind ``FRIDAY_ENABLE_PROTOCOLS``,
and adds no new dependencies. The public surface is:

* :class:`~friday.protocols.store.Protocol` / :class:`~friday.protocols.store.ProtocolStep`
  / :class:`~friday.protocols.store.SQLiteProtocolStore` — the typed models + the
  durable store.
* :class:`~friday.protocols.runner.ProtocolRunner` / its result models
  (:class:`~friday.protocols.runner.ProtocolResult` /
  :class:`~friday.protocols.runner.StepOutcome`) — the in-order executor over the
  registry that stops before an unconfirmed side-effecting step.
"""

from __future__ import annotations

from friday.protocols.runner import ProtocolResult, ProtocolRunner, StepOutcome
from friday.protocols.store import Protocol, ProtocolStep, SQLiteProtocolStore

__all__ = [
    "Protocol",
    "ProtocolResult",
    "ProtocolRunner",
    "ProtocolStep",
    "SQLiteProtocolStore",
    "StepOutcome",
]
