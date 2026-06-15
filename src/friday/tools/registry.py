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
from typing import Any, Protocol

from pydantic import ValidationError

from friday.errors import PermissionError
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.providers.llm import ToolSpec
from friday.tools.base import Tool, ToolError, ToolResult


class _HashAudit(Protocol):
    """The slice of a hash-chained ledger the registry appends to.

    Structurally satisfied by :class:`friday.broker.audit.HashChainedAudit`; kept
    as a local protocol so the registry imports nothing from the broker package
    (no cross-package coupling) and a bare registry needs no ledger at all.
    """

    def append(self, record: dict[str, Any]) -> Any:
        """Persist one hash-chained audit record and return the written entry."""
        ...


class ToolRegistry:
    """An in-process registry mapping tool names to :class:`Tool` instances.

    Observability is optional. When an :class:`~friday.observability.audit.AuditLog`
    and/or :class:`~friday.observability.metrics.Metrics` are injected, every
    :meth:`execute` records one redacted :class:`~friday.observability.audit.ToolCallAudit`
    row and bumps the ``tool_calls`` counter (build-spec §11). When they are absent
    (the default) the registry behaves exactly as before, so call sites — and unit
    tests — that construct it bare keep working unchanged.

    Args:
        audit: Optional tool-call audit log; rows are recorded on every execute.
        metrics: Optional counter set; ``tool_calls`` is incremented on every execute.
        hash_audit: Optional tamper-evident, hash-chained ledger. When injected,
            every executed tool call additionally appends ONE hash-chained record
            (the tamper-evident system-of-record), on top of the in-memory
            ``audit`` row. Purely additive: ``None`` (the default) keeps the bare
            registry behaviour, and the in-memory ``audit`` is never altered.
        correlation_id: The request id stamped onto audit rows. ``app.py`` rebuilds
            the registry's wiring per request (or sets this) so rows tie back to
            the turn's trace; defaults to ``"-"`` when unset.
    """

    def __init__(
        self,
        *,
        audit: AuditLog | None = None,
        metrics: Metrics | None = None,
        hash_audit: _HashAudit | None = None,
        correlation_id: str = "-",
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._audit = audit
        self._metrics = metrics
        self._hash_audit = hash_audit
        self._correlation_id = correlation_id

    def register(self, tool: Tool) -> None:
        """Register ``tool`` under its ``name``, replacing any prior entry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the registered tool named ``name`` or raise :class:`KeyError`."""
        return self._tools[name]

    def iter_tools(self) -> list[Tool]:
        """Return every registered :class:`Tool`, in registration order.

        The read-only enumeration the
        :class:`~friday.tools.capabilities.CapabilitiesTool` reflects over (it
        satisfies that tool's ``ToolLister`` structural contract). Never dispatches
        — it only lists what is registered — so a registry can describe its own
        capabilities without any caller reaching into its internals.
        """
        return list(self._tools.values())

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
        *,
        confirmed: bool = False,
    ) -> ToolResult:
        """Validate, gate side-effects, then dispatch the named tool.

        Order of checks matters and is part of the contract (build-spec §12):

        1. If ``name`` is not in ``allowed_tools`` -> raise
           :class:`friday.errors.PermissionError` (tool never runs).
        2. If ``name`` is not registered -> raise :class:`friday.errors.PermissionError`
           (an unknown tool is, by definition, not permitted).
        3. Validate ``raw_args`` against the tool's ``args_model``. On failure
           return ``ToolResult(ok=False, error=ToolError(code="bad_args"))``
           *without* invoking the tool.
        4. **Confirm-step.** If the tool is ``side_effecting`` *and* not
           ``idempotent`` *and* ``confirmed`` is ``False``, return
           ``ToolResult(ok=False, error=ToolError(code="confirmation_required"))``
           carrying ``data={"needs_confirmation": True, "tool": name}`` *without*
           invoking the tool. A read-only or idempotent tool, or any tool called
           with ``confirmed=True``, skips this gate.
        5. Otherwise invoke the tool with the validated args and return its
           :class:`ToolResult`.

        **Observability (build-spec §11).** A permission denial raises *before*
        the tool is ever considered, so it is not audited as a tool call. Every
        path that reaches a real tool decision — bad args, confirmation required,
        or an actual invocation — records one redacted
        :class:`~friday.observability.audit.ToolCallAudit` row and bumps the
        ``tool_calls`` metric (no-ops when neither store is wired).
        """
        if name not in allowed_tools:
            raise PermissionError(f"tool {name!r} is not in the allowed set")

        tool = self._tools.get(name)
        if tool is None:
            raise PermissionError(f"tool {name!r} is not registered")

        result = await self._dispatch(tool, name, raw_args, confirmed=confirmed)
        self._emit(name, raw_args, result)
        return result

    async def _dispatch(
        self,
        tool: Tool,
        name: str,
        raw_args: dict[str, object],
        *,
        confirmed: bool,
    ) -> ToolResult:
        """Validate, gate the confirm-step, then invoke ``tool`` (steps 3-5)."""
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

        if tool.side_effecting and not tool.idempotent and not confirmed:
            return ToolResult(
                ok=False,
                data={"needs_confirmation": True, "tool": name},
                error=ToolError(
                    code="confirmation_required",
                    message=(
                        f"tool {name!r} is side-effecting and not idempotent; "
                        "explicit confirmation is required before execution"
                    ),
                    retriable=False,
                ),
            )

        return await tool(args)

    def _emit(
        self, name: str, raw_args: dict[str, object], result: ToolResult
    ) -> None:
        """Record the audit row + metric for one executed tool call (if wired)."""
        if self._metrics is not None:
            self._metrics.inc_tool_calls()
        if self._audit is not None:
            self._audit.record(
                correlation_id=self._correlation_id,
                tool=name,
                args=raw_args,
                ok=result.ok,
                error_code=result.error.code if result.error is not None else None,
            )
        if self._hash_audit is not None:
            # Additive tamper-evident record. The ledger redacts sensitive-keyed
            # values itself before hashing/persisting, so passing ``raw_args`` is
            # safe (no credential reaches disk).
            self._hash_audit.append(
                {
                    "correlation_id": self._correlation_id,
                    "tool": name,
                    "ok": result.ok,
                    "error_code": (
                        result.error.code if result.error is not None else None
                    ),
                    "args": dict(raw_args),
                }
            )
