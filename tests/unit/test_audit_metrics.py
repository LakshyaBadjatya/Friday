"""Unit tests for the audit log + metrics counters (Phase 5, Stage 1).

Covers :class:`friday.observability.audit.AuditLog` and
:class:`friday.observability.metrics.Metrics`, plus their wiring into
:class:`friday.tools.registry.ToolRegistry` (a tool execute records exactly one
audit row with sensitive args redacted and bumps ``tool_calls``).
"""

from __future__ import annotations

from pydantic import BaseModel

from friday.logging import REDACTED
from friday.observability.audit import AuditLog, ToolCallAudit
from friday.observability.metrics import Metrics
from friday.tools.base import ToolResult
from friday.tools.registry import ToolRegistry


class _SearchArgs(BaseModel):
    query: str
    api_key: str | None = None


class _SpyTool:
    """A read-only tool that records whether it ran."""

    name = "spy"
    description = "spy tool"
    args_model = _SearchArgs
    required_permission = "spy"
    idempotent = True
    side_effecting = False

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, args: _SearchArgs) -> ToolResult:
        self.called = True
        return ToolResult(ok=True, data={"echo": args.query})


# --- AuditLog ------------------------------------------------------------- #


def _clock() -> float:
    return 42.0


def test_audit_record_redacts_sensitive_arg_keys() -> None:
    log = AuditLog(clock=_clock)
    log.record(
        correlation_id="cid-1",
        tool="web_search",
        args={"query": "vector dbs", "api_key": "nvapi-secret"},
        ok=True,
        error_code=None,
    )
    rows = log.recent(10)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, ToolCallAudit)
    assert row.correlation_id == "cid-1"
    assert row.tool == "web_search"
    assert row.ok is True
    assert row.error_code is None
    assert row.ts == 42.0
    # The plain arg survives; the sensitive key is redacted.
    assert row.args_redacted["query"] == "vector dbs"
    assert row.args_redacted["api_key"] == REDACTED


def test_audit_records_error_code_on_failure() -> None:
    log = AuditLog(clock=_clock)
    log.record(
        correlation_id="cid-2",
        tool="home",
        args={"device_id": "lamp"},
        ok=False,
        error_code="bad_args",
    )
    row = log.recent(10)[0]
    assert row.ok is False
    assert row.error_code == "bad_args"


def test_audit_ring_buffer_bounds_capacity() -> None:
    log = AuditLog(capacity=3)
    for i in range(5):
        log.record(
            correlation_id=f"cid-{i}",
            tool="t",
            args={},
            ok=True,
            error_code=None,
        )
    rows = log.recent(10)
    assert [r.correlation_id for r in rows] == ["cid-2", "cid-3", "cid-4"]


def test_audit_recent_respects_limit() -> None:
    log = AuditLog(capacity=10)
    for i in range(5):
        log.record(
            correlation_id=f"cid-{i}", tool="t", args={}, ok=True, error_code=None
        )
    rows = log.recent(2)
    assert [r.correlation_id for r in rows] == ["cid-3", "cid-4"]


# --- Metrics -------------------------------------------------------------- #


def test_metrics_counters_increment_and_snapshot() -> None:
    metrics = Metrics()
    metrics.inc_requests()
    metrics.inc_requests()
    metrics.inc_tool_calls()
    metrics.inc_errors()
    metrics.inc_mode("CONVERSATION")
    metrics.inc_mode("CONVERSATION")
    metrics.inc_mode("RESEARCH")

    snap = metrics.snapshot()
    assert snap["requests"] == 2
    assert snap["tool_calls"] == 1
    assert snap["errors"] == 1
    assert snap["by_mode"] == {"CONVERSATION": 2, "RESEARCH": 1}


def test_metrics_snapshot_is_a_copy() -> None:
    metrics = Metrics()
    metrics.inc_mode("CONVERSATION")
    snap = metrics.snapshot()
    # Mutating the snapshot must not corrupt internal state.
    snap["by_mode"]["CONVERSATION"] = 999
    snap["requests"] = 999
    assert metrics.snapshot()["by_mode"]["CONVERSATION"] == 1
    assert metrics.snapshot()["requests"] == 0


# --- Registry wiring ------------------------------------------------------ #


async def test_registry_without_observability_still_works() -> None:
    # Existing call sites construct the registry with no observability deps.
    registry = ToolRegistry()
    registry.register(_SpyTool())
    result = await registry.execute(
        "spy", {"query": "hi"}, allowed_tools={"spy"}
    )
    assert result.ok is True


async def test_registry_execute_records_one_audit_row_redacted() -> None:
    audit = AuditLog(clock=_clock)
    metrics = Metrics()
    registry = ToolRegistry(audit=audit, metrics=metrics, correlation_id="cid-exec")
    registry.register(_SpyTool())

    result = await registry.execute(
        "spy",
        {"query": "secrets please", "api_key": "nvapi-xyz"},
        allowed_tools={"spy"},
    )

    assert result.ok is True
    rows = audit.recent(10)
    assert len(rows) == 1
    row = rows[0]
    assert row.tool == "spy"
    assert row.correlation_id == "cid-exec"
    assert row.ok is True
    assert row.args_redacted["query"] == "secrets please"
    assert row.args_redacted["api_key"] == REDACTED
    # The tool-call counter advanced exactly once.
    assert metrics.snapshot()["tool_calls"] == 1


async def test_registry_records_audit_for_bad_args() -> None:
    audit = AuditLog(clock=_clock)
    metrics = Metrics()
    registry = ToolRegistry(audit=audit, metrics=metrics)
    registry.register(_SpyTool())

    # "query" missing -> validation fails; an audit row still lands (ok=False).
    result = await registry.execute("spy", {}, allowed_tools={"spy"})

    assert result.ok is False
    assert result.error is not None
    rows = audit.recent(10)
    assert len(rows) == 1
    assert rows[0].ok is False
    assert rows[0].error_code == "bad_args"
    assert metrics.snapshot()["tool_calls"] == 1
