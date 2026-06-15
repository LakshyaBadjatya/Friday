"""Unit tests for :mod:`friday.observability.tracing` (Phase 5, Stage 1).

The tracer is the per-request observability spine: every turn opens a
:class:`~friday.observability.tracing.Trace` whose :class:`Span` entries time the
route -> dispatch -> synth phases. Time is *injected* (a ``clock`` callable) so
these assertions are exact rather than flaky wall-clock comparisons.

Pinned here:

* a span context manager records start/end from the injected clock and the
  attributes passed to it;
* :meth:`Tracer.finish` closes the active trace and stamps its ``mode``;
* :meth:`Tracer.recent` returns traces newest-last, bounded by the ring buffer
  capacity (older traces are evicted, never the newest).
"""

from __future__ import annotations

from pathlib import Path

from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.observability.metrics import Metrics
from friday.observability.tracing import Span, Trace, Tracer
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.registry import ToolRegistry

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


class _FakeClock:
    """A deterministic clock: each call returns the next scripted float."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        # Saturate on the last value so an over-call never raises.
        tick = self._ticks[min(self._i, len(self._ticks) - 1)]
        self._i += 1
        return tick


def test_start_trace_stamps_started_from_clock() -> None:
    clock = _FakeClock([10.0, 11.0])
    tracer = Tracer(clock=clock)

    trace = tracer.start_trace("cid-1")

    assert isinstance(trace, Trace)
    assert trace.correlation_id == "cid-1"
    assert trace.started == 10.0
    assert trace.spans == []
    assert trace.mode is None


def test_span_records_timing_and_attrs() -> None:
    # start_trace consumes one tick; the span open/close consume two more.
    clock = _FakeClock([0.0, 1.0, 4.5])
    tracer = Tracer(clock=clock)
    tracer.start_trace("cid-2")

    with tracer.span("route", agent="router") as span:
        assert isinstance(span, Span)

    trace = tracer.finish()
    assert len(trace.spans) == 1
    recorded = trace.spans[0]
    assert recorded.name == "route"
    assert recorded.start == 1.0
    assert recorded.end == 4.5
    assert recorded.attrs == {"agent": "router"}


def test_multiple_spans_recorded_in_order() -> None:
    clock = _FakeClock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tracer = Tracer(clock=clock)
    tracer.start_trace("cid-3")

    for name in ("route", "dispatch", "synth"):
        with tracer.span(name):
            pass

    trace = tracer.finish()
    assert [s.name for s in trace.spans] == ["route", "dispatch", "synth"]
    # Each span's end is >= its start, and spans are time-ordered.
    for s in trace.spans:
        assert s.end is not None
        assert s.end >= s.start


def test_finish_stamps_mode_and_returns_trace() -> None:
    clock = _FakeClock([0.0])
    tracer = Tracer(clock=clock)
    trace_in = tracer.start_trace("cid-4")
    trace_in.mode = "CONVERSATION"

    finished = tracer.finish()

    assert finished is trace_in
    assert finished.mode == "CONVERSATION"


def test_finished_trace_appears_in_recent() -> None:
    clock = _FakeClock([0.0])
    tracer = Tracer(clock=clock)
    tracer.start_trace("cid-5")
    tracer.finish()

    recent = tracer.recent(10)
    assert len(recent) == 1
    assert recent[0].correlation_id == "cid-5"


def test_recent_ring_buffer_bounds_capacity() -> None:
    clock = _FakeClock([0.0])
    tracer = Tracer(clock=clock, capacity=3)

    for i in range(5):
        tracer.start_trace(f"cid-{i}")
        tracer.finish()

    recent = tracer.recent(10)
    # Only the last 3 survive; the two oldest were evicted.
    assert [t.correlation_id for t in recent] == ["cid-2", "cid-3", "cid-4"]


def test_recent_respects_limit_argument() -> None:
    clock = _FakeClock([0.0])
    tracer = Tracer(clock=clock, capacity=10)

    for i in range(5):
        tracer.start_trace(f"cid-{i}")
        tracer.finish()

    # limit caps the returned slice to the newest `limit` traces.
    recent = tracer.recent(2)
    assert [t.correlation_id for t in recent] == ["cid-3", "cid-4"]


def test_span_without_active_trace_is_noop() -> None:
    # A span opened before any trace started must not raise — it is simply
    # dropped (defensive: keeps emit points crash-free if mis-wired).
    clock = _FakeClock([0.0, 1.0])
    tracer = Tracer(clock=clock)
    with tracer.span("orphan"):
        pass
    assert tracer.recent(10) == []


# --- orchestrator wiring -------------------------------------------------- #


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


async def test_orchestrator_turn_emits_trace_with_expected_spans() -> None:
    # A plain conversation turn must open a trace whose spans cover the
    # route -> dispatch -> synth phases, stamped with the turn's mode, and the
    # request + by-mode metrics must advance.
    clock = _FakeClock([float(i) for i in range(20)])
    tracer = Tracer(clock=clock)
    metrics = Metrics()
    llm = FakeLLM(responses=[_resp("Four, Boss.")])
    orch = Orchestrator(
        llm=llm,
        registry=ToolRegistry(),
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        tracer=tracer,
        metrics=metrics,
    )

    out = await orch.handle(GraphState(session_id="s1", user_input="what's 2+2"))

    assert out.mode is Mode.CONVERSATION
    traces = tracer.recent(10)
    assert len(traces) == 1
    trace = traces[0]
    span_names = [s.name for s in trace.spans]
    assert span_names == ["route", "dispatch", "synth"]
    assert trace.mode == "CONVERSATION"
    snap = metrics.snapshot()
    assert snap["requests"] == 1
    assert snap["by_mode"]["CONVERSATION"] == 1
    assert snap["errors"] == 0


async def test_orchestrator_without_tracer_is_unchanged() -> None:
    # No tracer/metrics injected: the turn still completes (no-op defaults).
    llm = FakeLLM(responses=[_resp("Four, Boss.")])
    orch = Orchestrator(
        llm=llm,
        registry=ToolRegistry(),
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
    )
    out = await orch.handle(GraphState(session_id="s2", user_input="what's 2+2"))
    assert out.response is not None
