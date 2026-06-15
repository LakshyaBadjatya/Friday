"""The action broker: a fail-closed gate between intents and tool execution.

This package mediates every side-effecting (and read-only) tool call FRIDAY
makes. :class:`~friday.broker.broker.Broker` validates arguments, classifies
reversibility, enforces a deny-by-default permission gate, injects secrets at
call time (without ever surfacing them), executes via the tool registry, and
records the outcome in a tamper-evident
:class:`~friday.broker.audit.HashChainedAudit` ledger.

Both classes take all their collaborators as constructor parameters, so the
package imports nothing from :mod:`friday.config` or :mod:`friday.app`.
"""

from __future__ import annotations

from friday.broker.audit import HashChainedAudit
from friday.broker.broker import Broker

__all__ = ["Broker", "HashChainedAudit"]
