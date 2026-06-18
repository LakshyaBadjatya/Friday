# © Lakshya Badjatya — Author
"""Dry-run support: a broker shim that predicts tool calls without executing them.

:class:`DryRunBroker` has the same ``dispatch`` shape as the real
:class:`~friday.broker.broker.Broker`, but every dispatched call returns a
*predicted* successful result and is recorded instead of touching the world. The
Flow Engine swaps it in when ``run(..., simulate=True)`` so the owner can preview
exactly which tool steps would fire — and with what arguments — before arming a
flow for real, with a hard guarantee that no real side effect occurs.
"""

from __future__ import annotations

from typing import Any

from friday.tools.base import ToolResult


class DryRunBroker:
    """Wrap a real broker; predict side-effecting calls, delegate read-only ones.

    Args:
        inner: The real broker; read-only (non-side-effecting) calls are delegated
            to it unchanged.
        side_effecting: The set of tool names treated as side-effecting (predicted
            rather than executed). When a step is dispatched the engine passes
            ``side_effecting`` per call, so this is a conservative fallback set.
    """

    def __init__(self, inner: Any, side_effecting: frozenset[str] | None = None) -> None:
        self._inner = inner
        self._side_effecting = side_effecting or frozenset()
        self.predicted: list[dict[str, Any]] = []

    async def dispatch(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        *,
        allowed_tools: frozenset[str] | set[str],
        confirmed: bool = False,
        actor: str = "owner",
        channel: str = "chat",
    ) -> ToolResult:
        """Predict ``tool_name`` (record + simulated-ok) without executing it."""
        prediction = {"tool": tool_name, "args": dict(raw_args)}
        self.predicted.append(prediction)
        return ToolResult(ok=True, data={"simulated": True, **prediction})
