"""Unit tests for the defensive security tool + lockdown subgraph.

The security tool (``friday.tools.security``) exposes three defensive,
owner-scoped actions — ``revoke_tokens``, ``kill_sessions``, ``notify_owner`` —
each operating on FAKE owner-configured resources held in in-memory state and
each returning an :class:`AuditRecord` ``{step, ok, detail}``. They are
defensive-only: they never reach beyond the owner's own resources.

The lockdown subgraph (``friday.core.security.run_lockdown``) runs the three
steps IN ORDER and collects exactly one audit record per step. No LLM, no SDK —
``core`` stays clean (enforced separately by ``test_architecture.py``).
"""

from __future__ import annotations

from friday.core.security import run_lockdown
from friday.core.state import GraphState
from friday.tools.security import (
    AuditRecord,
    kill_sessions,
    notify_owner,
    revoke_tokens,
)

# Expected step order for a full lockdown run.
_EXPECTED_STEPS = ["revoke_tokens", "kill_sessions", "notify_owner"]


def _state() -> GraphState:
    return GraphState(session_id="sec-test", user_input="barn door procedure")


def test_revoke_tokens_returns_ok_audit_record() -> None:
    record = revoke_tokens()
    assert isinstance(record, AuditRecord)
    assert record.step == "revoke_tokens"
    assert record.ok is True
    assert record.detail  # non-empty human-readable detail


def test_kill_sessions_returns_ok_audit_record() -> None:
    record = kill_sessions()
    assert isinstance(record, AuditRecord)
    assert record.step == "kill_sessions"
    assert record.ok is True
    assert record.detail


def test_notify_owner_returns_ok_audit_record() -> None:
    record = notify_owner()
    assert isinstance(record, AuditRecord)
    assert record.step == "notify_owner"
    assert record.ok is True
    assert record.detail


def test_audit_record_is_a_clean_value_object() -> None:
    # No wall-clock fields baked in; it round-trips losslessly.
    record = AuditRecord(step="x", ok=True, detail="y")
    assert record.model_dump() == {"step": "x", "ok": True, "detail": "y"}


async def test_run_lockdown_produces_three_ordered_ok_records() -> None:
    records = await run_lockdown(_state())
    assert len(records) == 3
    assert [r.step for r in records] == _EXPECTED_STEPS
    assert all(isinstance(r, AuditRecord) for r in records)
    assert all(r.ok for r in records)


async def test_run_lockdown_is_deterministic_and_order_stable() -> None:
    first = await run_lockdown(_state())
    second = await run_lockdown(_state())
    assert [r.step for r in first] == [r.step for r in second] == _EXPECTED_STEPS
