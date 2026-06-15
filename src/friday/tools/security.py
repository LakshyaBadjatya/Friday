"""Defensive, owner-scoped security actions for the lockdown ("barn door") path.

This module exposes the three primitive steps of FRIDAY's defensive lockdown
(build-spec §9.9). Each acts ONLY on FAKE, owner-configured resources held in
in-memory state — there is no real credential store, session manager, or
notifier here — and each returns an :class:`AuditRecord` describing what it did.

**Defensive-only by construction.** These functions revoke the *owner's own*
tokens, terminate the *owner's own* sessions, and notify the *owner*. They never
reach beyond the owner's resources: there is no target/host/account parameter to
point them at someone else, and the in-memory fakes model exactly one tenant —
the owner. This is the whole point of the "barn door" procedure: shut the
owner's own doors fast, then tell the owner.

**No wall-clock here.** :class:`AuditRecord` deliberately carries no timestamp.
Per the plan, time-stamping is the caller's job (so tests stay deterministic and
don't depend on ``datetime.now``). The lockdown subgraph in
``friday.core.security`` simply collects these records in order.
"""

from __future__ import annotations

from pydantic import BaseModel


class AuditRecord(BaseModel):
    """One line of the lockdown audit trail.

    Attributes:
        step: The lockdown step that produced this record (e.g.
            ``"revoke_tokens"``). Stable identifiers, asserted in order by the
            subgraph's tests.
        ok: Whether the step completed successfully against the owner's fakes.
        detail: A short, human-readable summary of what happened (counts, names
            of the owner-scoped resources touched). Never empty.
    """

    step: str
    ok: bool
    detail: str


# --------------------------------------------------------------------------- #
# Fake, owner-configured resources (in-memory). These stand in for a real
# credential store / session manager / notifier and exist only so the lockdown
# path is exercisable end-to-end offline. They model a SINGLE tenant — the owner
# — so there is structurally no way to point these actions at anyone else.
# --------------------------------------------------------------------------- #

# The owner's currently-issued access tokens (opaque identifiers).
_OWNER_TOKENS: tuple[str, ...] = ("owner-api", "owner-mobile", "owner-cli")
# The owner's currently-active session identifiers.
_OWNER_SESSIONS: tuple[str, ...] = ("sess-web", "sess-desktop")
# Where the owner is notified (a fake channel handle, not a real address).
_OWNER_NOTIFY_CHANNEL = "owner-secure-inbox"


def revoke_tokens() -> AuditRecord:
    """Revoke the owner's own outstanding access tokens (fake, in-memory).

    Defensive: only the owner's configured tokens are touched; there is no
    parameter to aim this elsewhere.
    """
    count = len(_OWNER_TOKENS)
    detail = (
        f"revoked {count} owner token(s): {', '.join(_OWNER_TOKENS)}"
        if count
        else "no owner tokens outstanding"
    )
    return AuditRecord(step="revoke_tokens", ok=True, detail=detail)


def kill_sessions() -> AuditRecord:
    """Terminate the owner's own active sessions (fake, in-memory).

    Defensive: scoped to the owner's configured sessions only.
    """
    count = len(_OWNER_SESSIONS)
    detail = (
        f"killed {count} owner session(s): {', '.join(_OWNER_SESSIONS)}"
        if count
        else "no owner sessions active"
    )
    return AuditRecord(step="kill_sessions", ok=True, detail=detail)


def notify_owner() -> AuditRecord:
    """Notify the owner that a lockdown occurred (fake, in-memory channel).

    Defensive: addresses the owner's own configured channel; it does not, and
    cannot, message a third party.
    """
    detail = f"notified owner via {_OWNER_NOTIFY_CHANNEL}"
    return AuditRecord(step="notify_owner", ok=True, detail=detail)
