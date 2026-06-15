"""Observability primitives: per-request tracing, a tool-call audit, and metrics.

Phase 5 (build-spec §11, §3) adds a thin, in-process observability layer that the
orchestrator and tool registry emit into and the admin API reads back:

* :mod:`friday.observability.tracing` — a :class:`~friday.observability.tracing.Tracer`
  that opens a :class:`~friday.observability.tracing.Trace` per turn with timed
  :class:`~friday.observability.tracing.Span` entries, backed by a bounded ring
  buffer. Time is injected (a ``clock`` callable) so traces are deterministic in
  tests — no direct wall-clock on a tested path.
* :mod:`friday.observability.audit` — an
  :class:`~friday.observability.audit.AuditLog` of
  :class:`~friday.observability.audit.ToolCallAudit` rows (sensitive args redacted
  with the same key-set as :mod:`friday.logging`).
* :mod:`friday.observability.metrics` — simple
  :class:`~friday.observability.metrics.Metrics` counters with a
  :meth:`~friday.observability.metrics.Metrics.snapshot`.

Everything here is pure, dependency-light, and importable from business logic;
it never reaches the network or an LLM SDK.
"""

from __future__ import annotations

from friday.observability.audit import AuditLog, ToolCallAudit
from friday.observability.metrics import Metrics
from friday.observability.tracing import Span, Trace, Tracer

__all__ = [
    "AuditLog",
    "Metrics",
    "Span",
    "ToolCallAudit",
    "Trace",
    "Tracer",
]
