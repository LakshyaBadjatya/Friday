"""The tool-call audit trail (build-spec §11).

Every tool the registry executes lands one :class:`ToolCallAudit` row in an
in-memory :class:`AuditLog`: which tool, with which (redacted) args, whether it
succeeded, and — on failure — the result's error code. The ``GET /admin/audit``
view reads :meth:`AuditLog.recent` back.

**Redaction reuses the logging key-set.** Argument values whose key looks like a
secret (``api_key``, ``token``, ``secret``, ``password``, ``authorization`` —
the same substrings :mod:`friday.logging` redacts) are replaced with the shared
:data:`friday.logging.REDACTED` sentinel, so a credential passed as a tool arg is
never persisted in the audit, mirroring the log redaction exactly.

Time is injected (a ``clock`` callable) so audit timestamps are deterministic on
a tested path — no direct wall-clock here.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from friday.logging import REDACTED, _is_sensitive  # shared sensitive-key set

_DEFAULT_CAPACITY = 512


def _redact_args(args: dict[str, object]) -> dict[str, Any]:
    """Return a copy of ``args`` with sensitive-keyed values replaced.

    Uses the same case-insensitive substring predicate as :mod:`friday.logging`,
    so the audit and the structured logs redact exactly the same keys.
    """
    return {key: (REDACTED if _is_sensitive(key) else value) for key, value in args.items()}


class ToolCallAudit(BaseModel):
    """One audited tool invocation.

    Attributes:
        correlation_id: The request the call belonged to (ties it to the trace
            and the log lines).
        tool: The tool's registered name.
        args_redacted: The call's arguments with sensitive values redacted.
        ok: Whether the call returned a successful :class:`~friday.tools.base.ToolResult`.
        error_code: The result's ``error.code`` on a handled failure, else ``None``.
        ts: The clock reading when the row was recorded.
    """

    correlation_id: str
    tool: str
    args_redacted: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error_code: str | None = None
    ts: float


class AuditLog:
    """An in-memory, bounded ring buffer of :class:`ToolCallAudit` rows.

    Args:
        clock: A zero-arg callable returning a ``float`` timestamp. Injected for
            deterministic tests; defaults to :func:`time.time`.
        capacity: Maximum rows retained; older rows are evicted FIFO.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        capacity: int = _DEFAULT_CAPACITY,
    ) -> None:
        self._clock = clock
        self._rows: deque[ToolCallAudit] = deque(maxlen=capacity)

    def record(
        self,
        *,
        correlation_id: str,
        tool: str,
        args: dict[str, object],
        ok: bool,
        error_code: str | None,
    ) -> ToolCallAudit:
        """Append a redacted audit row and return it."""
        row = ToolCallAudit(
            correlation_id=correlation_id,
            tool=tool,
            args_redacted=_redact_args(args),
            ok=ok,
            error_code=error_code,
            ts=self._clock(),
        )
        self._rows.append(row)
        return row

    def recent(self, limit: int = 50) -> list[ToolCallAudit]:
        """Return up to ``limit`` most-recent rows, oldest-first."""
        rows = list(self._rows)
        if limit >= 0:
            rows = rows[-limit:] if limit else []
        return rows
