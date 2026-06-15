"""Typed tool registry with permission enforcement and pre-execution validation.

The registry is the single place tools are dispatched. It:

* generates :class:`~friday.providers.llm.ToolSpec` entries straight from each
  tool's ``args_model.model_json_schema()`` so the schema the LLM sees can never
  drift from the schema actually enforced;
* enforces the per-call allow-list — a tool not in ``allowed_tools`` raises
  :class:`friday.errors.PermissionError` and is *never* invoked; and
* validates raw arguments against the tool's ``args_model`` *before* calling it —
  invalid arguments return ``ToolResult(ok=False, error=ToolError(code="bad_args"))``
  and the tool body never runs.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from friday.errors import PermissionError
from friday.providers.llm import ToolSpec
from friday.tools.base import Tool, ToolError, ToolResult


class ToolRegistry:
    """An in-process registry mapping tool names to :class:`Tool` instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register ``tool`` under its ``name``, replacing any prior entry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the registered tool named ``name`` or raise :class:`KeyError`."""
        return self._tools[name]

    def spec_for(self, names: Iterable[str]) -> list[ToolSpec]:
        """Return :class:`ToolSpec` entries for ``names``.

        Each spec's ``parameters`` is the tool's ``args_model`` JSON schema, so
        the declared interface is generated from the same model that validates
        calls. Raises :class:`KeyError` for an unknown name.
        """
        specs: list[ToolSpec] = []
        for name in names:
            tool = self._tools[name]
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.args_model.model_json_schema(),
                )
            )
        return specs

    async def execute(
        self,
        name: str,
        raw_args: dict[str, object],
        allowed_tools: frozenset[str] | set[str],
    ) -> ToolResult:
        """Validate, then dispatch the named tool.

        Order of checks matters and is part of the contract:

        1. If ``name`` is not in ``allowed_tools`` -> raise
           :class:`friday.errors.PermissionError` (tool never runs).
        2. If ``name`` is not registered -> raise :class:`friday.errors.PermissionError`
           (an unknown tool is, by definition, not permitted).
        3. Validate ``raw_args`` against the tool's ``args_model``. On failure
           return ``ToolResult(ok=False, error=ToolError(code="bad_args"))``
           *without* invoking the tool.
        4. Otherwise invoke the tool with the validated args and return its
           :class:`ToolResult`.
        """
        if name not in allowed_tools:
            raise PermissionError(f"tool {name!r} is not in the allowed set")

        tool = self._tools.get(name)
        if tool is None:
            raise PermissionError(f"tool {name!r} is not registered")

        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="bad_args",
                    message=f"invalid arguments for tool {name!r}: {exc.errors()}",
                    retriable=False,
                ),
            )

        return await tool(args)
