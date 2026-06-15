"""The defensive lockdown ("barn door") subgraph.

When the router classifies a turn into :class:`~friday.core.state.Mode.SECURITY_LOCKDOWN`
the orchestrator hands off to this module rather than to a chatty agent. The
lockdown is a fixed, ordered, defensive-only procedure (build-spec §9.9):

1. ``revoke_tokens`` — drop the owner's own outstanding access tokens.
2. ``kill_sessions`` — terminate the owner's own active sessions.
3. ``notify_owner`` — tell the owner a lockdown ran.

Each step returns an :class:`~friday.tools.security.AuditRecord`; this function
collects exactly one record per step, in that order, and returns the audit
trail for the orchestrator to report. There is **no LLM and no SDK** here — the
steps act on fake, owner-scoped in-memory resources — so ``core`` stays clean
(enforced by ``tests/unit/test_architecture.py``).

The ``state`` argument is accepted so this slots into the mode-loop alongside
the other node functions; the procedure itself is deterministic and does not
branch on conversation content (a lockdown is a lockdown).
"""

from __future__ import annotations

from friday.core.state import GraphState
from friday.tools.security import (
    AuditRecord,
    kill_sessions,
    notify_owner,
    revoke_tokens,
)

# The ordered lockdown steps. Order is load-bearing and asserted by tests:
# revoke first (cut credential reuse), then kill live sessions, then notify.
_LOCKDOWN_STEPS = (revoke_tokens, kill_sessions, notify_owner)


async def run_lockdown(state: GraphState) -> list[AuditRecord]:
    """Run the defensive lockdown steps in order, returning the audit trail.

    Args:
        state: The current graph state. Accepted for node-API uniformity; the
            lockdown procedure is fixed and does not branch on it.

    Returns:
        Exactly one :class:`AuditRecord` per step, in the order
        ``revoke_tokens`` -> ``kill_sessions`` -> ``notify_owner``.
    """
    return [step() for step in _LOCKDOWN_STEPS]
