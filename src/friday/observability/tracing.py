"""Per-request tracing with an injectable clock (build-spec §11).

A :class:`Tracer` opens one :class:`Trace` per turn; within it, :meth:`Tracer.span`
is a context manager that times a phase (route, dispatch, synth) and records its
attributes. :meth:`Tracer.finish` closes the active trace and files it in a
bounded ring buffer that :meth:`Tracer.recent` reads back for the admin API.

**Time is injected.** The tracer takes a ``clock`` callable returning a ``float``
(monotonic seconds in production, a scripted sequence in tests) and never calls
the wall clock itself, so span timings — and therefore the gate's trace
assertions — are deterministic offline.

The data classes are pydantic models so a trace round-trips to JSON for the
``GET /admin/traces`` view without bespoke serialization.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field

# Default ring-buffer depth: enough to inspect recent activity from the
# dashboard without growing without bound in a long-lived process.
_DEFAULT_CAPACITY = 256


class Span(BaseModel):
    """One timed phase within a :class:`Trace`.

    ``start`` and ``end`` are clock readings (the injected ``clock`` callable);
    ``end`` is ``None`` only while the span is still open. ``attrs`` carries any
    structured context the emit point attached (e.g. the chosen agent or mode).
    """

    name: str
    start: float
    end: float | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """The trace of a single turn: an ordered list of timed :class:`Span` s.

    ``correlation_id`` ties the trace to the request's log lines and audit rows.
    ``mode`` is stamped by the orchestrator once the turn's :class:`Mode` is
    known (left ``None`` for a trace that never reached routing).
    """

    correlation_id: str
    spans: list[Span] = Field(default_factory=list)
    started: float
    mode: str | None = None


class Tracer:
    """Opens, times, and retains traces for the requests it observes.

    Args:
        clock: A zero-arg callable returning a ``float`` reading. Injected so
            tests are deterministic; defaults to :func:`time.monotonic`.
        capacity: Maximum number of finished traces kept in the ring buffer.

    A :class:`Tracer` instance threads a single *active* trace at a time (one
    request per tracer, matching the per-request wiring in ``app.py``). Opening a
    span without an active trace is a safe no-op rather than an error, so a
    mis-wired emit point degrades gracefully instead of crashing a turn.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        capacity: int = _DEFAULT_CAPACITY,
    ) -> None:
        self._clock = clock
        self._traces: deque[Trace] = deque(maxlen=capacity)
        self._active: Trace | None = None

    def start_trace(self, correlation_id: str) -> Trace:
        """Open a new active trace stamped with the current clock reading."""
        trace = Trace(correlation_id=correlation_id, started=self._clock())
        self._active = trace
        return trace

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span | None]:
        """Time a ``name`` phase, recording it on the active trace.

        Yields the :class:`Span` so the caller may inspect it. If no trace is
        active the body still runs but nothing is recorded (yields ``None``),
        keeping emit points crash-free even if mis-wired.
        """
        active = self._active
        if active is None:
            yield None
            return
        span = Span(name=name, start=self._clock(), attrs=dict(attrs))
        try:
            yield span
        finally:
            span.end = self._clock()
            active.spans.append(span)

    def finish(self) -> Trace:
        """Close the active trace, file it in the ring buffer, and return it.

        Raises :class:`RuntimeError` if there is no active trace — a programmer
        error, since :meth:`finish` always pairs with a prior :meth:`start_trace`.
        """
        active = self._active
        if active is None:  # pragma: no cover - defensive; paired with start_trace
            raise RuntimeError("finish() called with no active trace")
        self._traces.append(active)
        self._active = None
        return active

    def recent(self, limit: int = 50) -> list[Trace]:
        """Return up to ``limit`` most-recent finished traces, oldest-first.

        The ring buffer already bounds total retention; ``limit`` further caps
        the returned slice to the newest ``limit`` traces.
        """
        traces = list(self._traces)
        if limit >= 0:
            traces = traces[-limit:] if limit else []
        return traces
