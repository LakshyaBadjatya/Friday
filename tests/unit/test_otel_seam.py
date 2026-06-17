# © Lakshya Badjatya — Author
"""Unit tests for the OpenTelemetry trace-export seam (flag-gated, lazy)."""

from __future__ import annotations

from friday.config import Settings
from friday.observability.otel import (
    OTelTraceExporter,
    RecordingTraceExporter,
    build_trace_exporter,
)
from friday.observability.tracing import Tracer


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", **overrides)  # type: ignore[arg-type]


def test_builder_returns_none_when_flag_off() -> None:
    assert build_trace_exporter(_settings()) is None


def test_builder_returns_exporter_when_flag_on() -> None:
    exporter = build_trace_exporter(_settings(enable_otel=True))
    assert isinstance(exporter, OTelTraceExporter)


def test_otel_exporter_degrades_when_sdk_missing() -> None:
    # The opentelemetry SDK is not installed in this env, so construction must
    # NOT raise and export() must be a safe no-op (degraded mode).
    exporter = OTelTraceExporter("http://localhost:4318/v1/traces")
    tracer = Tracer()
    tracer.start_trace("c1")
    with tracer.span("route"):
        pass
    trace = tracer.finish()
    exporter.export(trace)  # no exception


def test_tracer_forwards_finished_traces_to_exporter() -> None:
    recorder = RecordingTraceExporter()
    tracer = Tracer(exporter=recorder)
    tracer.start_trace("corr-1")
    with tracer.span("route"):
        pass
    tracer.finish()
    assert len(recorder.exported) == 1
    assert recorder.exported[0].correlation_id == "corr-1"
    assert [s.name for s in recorder.exported[0].spans] == ["route"]


def test_tracer_without_exporter_still_works() -> None:
    tracer = Tracer()
    tracer.start_trace("c")
    finished = tracer.finish()
    assert finished.correlation_id == "c"


def test_raising_exporter_never_breaks_finish() -> None:
    class _Boom:
        def export(self, trace: object) -> None:
            raise RuntimeError("collector down")

    tracer = Tracer(exporter=_Boom())
    tracer.start_trace("c")
    # finish() swallows the exporter error and still returns the trace.
    assert tracer.finish().correlation_id == "c"
