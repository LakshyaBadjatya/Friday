# © Lakshya Badjatya — Author
"""OpenTelemetry export seam — ship finished traces to an OTLP collector.

The in-process :class:`~friday.observability.tracing.Tracer` always keeps its
ring buffer for ``GET /admin/traces``; this seam *additionally* forwards each
finished trace to an OTLP/HTTP collector when ``enable_otel`` is set. It is the
one place the optional ``opentelemetry`` SDK is touched, and it is **lazy** — the
SDK is imported only when an :class:`OTelTraceExporter` is constructed, so the
offline default never needs it installed.

Fail-soft by construction: if the SDK isn't installed (or the collector can't be
reached) the exporter degrades to a logged warning and a no-op ``export`` — a
turn is never broken by telemetry. :class:`RecordingTraceExporter` is the
in-memory test double / reference implementation; :func:`build_trace_exporter`
returns ``None`` unless the flag is on, so wiring stays a one-liner.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from friday.observability.tracing import Trace

if TYPE_CHECKING:
    from friday.config import Settings

logger = logging.getLogger("friday.observability.otel")


@runtime_checkable
class TraceExporter(Protocol):
    """Anything that can ship a finished :class:`Trace` somewhere."""

    def export(self, trace: Trace) -> None:
        """Export one finished trace. Must never raise into the caller."""
        ...


class RecordingTraceExporter:
    """An in-memory exporter — the test double and a reference implementation."""

    def __init__(self) -> None:
        self.exported: list[Trace] = []

    def export(self, trace: Trace) -> None:
        """Record the trace in-memory."""
        self.exported.append(trace)


class OTelTraceExporter:
    """Forward finished traces to an OTLP/HTTP collector via the lazy SDK.

    The ``opentelemetry`` SDK is imported in ``__init__``; if it is missing the
    exporter logs once and enters a degraded no-op mode (``_tracer is None``) so
    boot never fails on a missing optional dependency. Each of our :class:`Trace`
    spans is emitted as one OTel span; our monotonic clock readings are carried as
    attributes (``start``/``end``/``duration_ms``) since they are not wall-clock.
    """

    def __init__(self, endpoint: str, service_name: str = "friday") -> None:
        self._endpoint = endpoint
        self._service_name = service_name
        self._tracer = None
        try:
            self._tracer = _build_otel_tracer(endpoint, service_name)
        except ImportError:
            logger.warning(
                "enable_otel is set but the opentelemetry SDK is not installed; "
                "trace export is disabled (install the OTel extras to enable it)"
            )

    def export(self, trace: Trace) -> None:
        """Emit ``trace`` as OTel spans; a no-op when the SDK is unavailable."""
        if self._tracer is None:
            return
        try:
            for span in trace.spans:
                duration_ms = (
                    int((span.end - span.start) * 1000) if span.end is not None else 0
                )
                otel_span = self._tracer.start_span(span.name)
                otel_span.set_attribute("friday.correlation_id", trace.correlation_id)
                otel_span.set_attribute("friday.mode", trace.mode or "")
                otel_span.set_attribute("friday.start", span.start)
                otel_span.set_attribute("friday.duration_ms", duration_ms)
                for key, value in span.attrs.items():
                    otel_span.set_attribute(f"friday.attr.{key}", str(value))
                otel_span.end()
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a turn
            logger.warning("OTel export failed (continuing): %s", exc)


def _build_otel_tracer(endpoint: str, service_name: str):  # type: ignore[no-untyped-def]
    """Lazy-build an OTel tracer wired to an OTLP/HTTP span exporter.

    Isolated so the heavy imports happen only when ``enable_otel`` is on. Raises
    :class:`ImportError` (caught by the caller) when the SDK is not installed.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]  # noqa: E501
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
        BatchSpanProcessor,
    )

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return provider.get_tracer("friday")


def build_trace_exporter(settings: Settings) -> TraceExporter | None:
    """Return an OTLP exporter when ``enable_otel`` is on, else ``None``.

    ``None`` means "no export" — the :class:`Tracer` keeps only its in-process
    ring buffer, which is the offline default.
    """
    if not settings.enable_otel:
        return None
    return OTelTraceExporter(settings.otel_endpoint, settings.otel_service_name)
