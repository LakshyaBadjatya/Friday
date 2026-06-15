"""Typed tool boundary: the :class:`Tool` protocol and its result payloads.

This module defines the contract every FRIDAY tool implements and the typed
values a tool returns.

**Naming note (deliberate):** there are two ``ToolError`` concepts in the
codebase and they are intentionally kept separate:

* :class:`friday.errors.ToolError` is an *exception* (a ``FridayError``
  subclass) — raised when something goes wrong control-flow-wise.
* :class:`friday.tools.base.ToolError` (this module) is a *typed result
  payload* — a pydantic model carried inside a :class:`ToolResult` to describe a
  failed-but-handled tool outcome (``code`` / ``message`` / ``retriable``).

Tools return ``ToolResult(ok=False, error=ToolError(...))`` for expected,
recoverable failures and only raise the exception family for programmer/permission
errors. The registry (``friday.tools.registry``) imports the *payload* from here
and the *exception* from ``friday.errors``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """Typed payload describing a handled tool failure.

    Distinct from the :class:`friday.errors.ToolError` exception; see the module
    docstring. ``retriable`` signals whether re-invoking the tool unchanged could
    plausibly succeed (e.g. a transient network blip) versus a deterministic
    failure (e.g. bad arguments) the caller should not blindly retry.
    """

    code: str
    message: str
    retriable: bool = False


class ToolResult(BaseModel):
    """The normalized result of a tool invocation.

    ``ok`` is the single source of truth for success. On success ``data`` carries
    the payload and ``error`` is ``None``; on a handled failure ``ok`` is
    ``False`` and ``error`` describes why.
    """

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None


@runtime_checkable
class Tool(Protocol):
    """Structural contract every FRIDAY tool implements.

    Attributes describe the tool to the registry and the permission system;
    ``__call__`` performs the work. ``args`` is an instance of ``args_model``
    (already validated by the registry) and the return is always a
    :class:`ToolResult` — tools surface expected failures as
    ``ToolResult(ok=False, error=...)`` rather than raising.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    required_permission: str
    idempotent: bool
    side_effecting: bool

    async def __call__(self, args: Any) -> ToolResult:
        """Execute the tool against validated ``args`` and return a result."""
        ...
