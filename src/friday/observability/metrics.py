"""Process-lifetime metric counters (build-spec §3 observability).

A single :class:`Metrics` instance accumulates a handful of monotonic counters —
total requests, total tool calls, total errors, and a per-:class:`Mode` request
breakdown — that the ``GET /admin/metrics`` view surfaces via
:meth:`Metrics.snapshot`. These are deliberately simple in-process counters (no
external metrics backend) sufficient for the local-first dashboard.

:meth:`Metrics.snapshot` returns a deep-enough copy that mutating the result can
never corrupt the live counters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class Metrics:
    """Mutable, in-process counters with a JSON-able :meth:`snapshot`."""

    def __init__(self) -> None:
        self._requests = 0
        self._tool_calls = 0
        self._errors = 0
        self._by_mode: defaultdict[str, int] = defaultdict(int)

    def inc_requests(self, n: int = 1) -> None:
        """Increment the total-requests counter."""
        self._requests += n

    def inc_tool_calls(self, n: int = 1) -> None:
        """Increment the total tool-call counter."""
        self._tool_calls += n

    def inc_errors(self, n: int = 1) -> None:
        """Increment the total-errors counter."""
        self._errors += n

    def inc_mode(self, mode: str, n: int = 1) -> None:
        """Increment the per-mode request counter for ``mode``."""
        self._by_mode[mode] += n

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current counters.

        The returned ``by_mode`` mapping is a fresh ``dict``, so a caller
        mutating the snapshot (e.g. shaping it for a response) cannot corrupt the
        live counters.
        """
        return {
            "requests": self._requests,
            "tool_calls": self._tool_calls,
            "errors": self._errors,
            "by_mode": dict(self._by_mode),
        }
