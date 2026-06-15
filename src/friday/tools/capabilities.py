"""Capabilities tool: a structured, self-describing map of available tools.

:class:`CapabilitiesTool` answers "what can you do?" by reading the injected tool
registry and returning, for every registered tool, its ``name``, ``description``,
and ``side_effecting`` flag (plus ``required_permission`` and ``idempotent`` for
completeness). The agent can surface this to the user or feed it back to itself
when planning, without hard-coding a tool list anywhere.

Dependency injection only: the registry arrives through the constructor, so this
module imports neither :mod:`friday.config` nor :mod:`friday.app`. To stay fully
decoupled from the concrete :class:`~friday.tools.registry.ToolRegistry` (and to
be unit-testable with a fake), the tool depends on a minimal **structural**
contract, :class:`ToolLister`, that yields the registered
:class:`~friday.tools.base.Tool` instances. The integration pass injects a
registry (or thin adapter) that satisfies it.

The tool is read-only (``side_effecting=False``, ``idempotent=True``) — it only
reflects over what is registered and never invokes anything.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from friday.tools.base import Tool, ToolResult

logger = logging.getLogger("friday.tools.capabilities")


@runtime_checkable
class ToolLister(Protocol):
    """Structural contract for a registry that can enumerate its tools.

    The capabilities tool needs only to *iterate* the registered
    :class:`~friday.tools.base.Tool` instances; it never dispatches through the
    registry. Anything exposing ``iter_tools()`` (returning an iterable of tools)
    satisfies this — the concrete registry, a thin adapter, or a test fake.
    """

    def iter_tools(self) -> list[Tool]: ...


class CapabilitiesArgs(BaseModel):
    """Arguments for :class:`CapabilitiesTool`.

    The tool takes no inputs — it always reports the full registered tool set —
    so the args model is intentionally empty. An empty pydantic model still gives
    the registry a real JSON schema to validate ``{}`` against.
    """


class CapabilitiesTool:
    """Report a structured map of every registered tool's capabilities.

    Args:
        registry: Any object exposing ``iter_tools()`` (see :class:`ToolLister`).
            Read-only; the tool only reflects over the registered tools and never
            dispatches through the registry.
    """

    name = "capabilities"
    description = (
        "List the assistant's available tools with their descriptions and "
        "whether each has external side-effects."
    )
    args_model = CapabilitiesArgs
    required_permission = "capabilities"
    idempotent = True
    side_effecting = False

    def __init__(self, registry: ToolLister) -> None:
        self._registry = registry

    async def __call__(self, args: Any) -> ToolResult:
        """Return ``{tools: [...], count: N}`` describing every registered tool."""
        if not isinstance(args, CapabilitiesArgs):
            args = CapabilitiesArgs.model_validate(args)
        tools = list(self._registry.iter_tools())
        entries = [self._describe(tool) for tool in tools]
        # Deterministic order so the same registry always yields the same map.
        entries.sort(key=lambda entry: entry["name"])
        data: dict[str, Any] = {"tools": entries, "count": len(entries)}
        logger.info("capabilities reported count=%d", len(entries))
        return ToolResult(ok=True, data=data, error=None)

    @staticmethod
    def _describe(tool: Tool) -> dict[str, Any]:
        """Serialize one tool's public capability fields to a plain dict."""
        return {
            "name": tool.name,
            "description": tool.description,
            "side_effecting": bool(tool.side_effecting),
            "idempotent": bool(tool.idempotent),
            "required_permission": tool.required_permission,
        }
