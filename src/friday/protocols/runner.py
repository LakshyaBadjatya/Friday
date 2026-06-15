"""The protocol runner: execute a named routine's steps through the registry.

A :class:`ProtocolRunner` runs a :class:`~friday.protocols.store.Protocol`'s steps
*in order* through the shared :class:`~friday.tools.registry.ToolRegistry`, so a
protocol can only ever invoke already-registered tools (no arbitrary execution)
and every call passes the registry's permission gate, argument validation, and
confirm-step.

The confirm-step is honored end-to-end. A side-effecting, non-idempotent step run
with ``confirmed=False`` returns a ``ToolResult`` carrying
``data["needs_confirmation"]`` *without* executing — the runner treats that as a
pause: it stops the run before the side-effecting work (``ran=False``,
``needs_confirmation=True``) and reports the steps so far, running NO later steps.
A confirming follow-up (``confirmed=True``) runs every step. Any step that returns
``ok=False`` (a tool failure or a permission denial) also stops the run and is
reported, so a protocol never silently half-completes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from friday.errors import PermissionError as FridayPermissionError
from friday.protocols.store import Protocol
from friday.tools.registry import ToolRegistry


class StepOutcome(BaseModel):
    """The result of attempting a single protocol step.

    ``ok`` is ``True`` only when the step's tool ran and returned success.
    ``needs_confirmation`` is set when the step paused on the confirm-step
    (side-effecting + unconfirmed): the tool did NOT execute. ``error`` carries
    the failure code on a tool failure / permission denial, else ``None``.
    """

    tool: str
    ok: bool
    error: str | None = None
    needs_confirmation: bool = False


class ProtocolResult(BaseModel):
    """The aggregate result of running a protocol.

    ``ran`` is ``True`` only when every step completed successfully. ``needs_confirmation``
    is set when the run paused on a side-effecting step awaiting the owner's
    confirmation (the side-effecting step and any later steps did NOT run).
    ``steps`` reports each step attempted, in order, including the paused/failed one.
    """

    protocol: str
    ran: bool
    needs_confirmation: bool = False
    steps: list[StepOutcome] = Field(default_factory=list)


class ProtocolRunner:
    """Execute a protocol's steps in order through the shared tool registry.

    Args:
        registry: The shared :class:`~friday.tools.registry.ToolRegistry`; every
            step dispatches through its ``execute`` so permission/validation/the
            confirm-step all apply unchanged.
        allowed_tools: The per-call allow-list passed to ``execute`` — the set of
            tool names a protocol may invoke (the app wires this to the registered
            tool names). A step naming a tool outside this set is denied by the
            registry and reported as a failed step.
    """

    def __init__(
        self, registry: ToolRegistry, allowed_tools: frozenset[str] | set[str]
    ) -> None:
        self._registry = registry
        self._allowed_tools = frozenset(allowed_tools)

    async def run(
        self, protocol: Protocol, *, confirmed: bool = False
    ) -> ProtocolResult:
        """Run ``protocol``'s steps in order; stop on a pause or a failure.

        Each step is dispatched via ``registry.execute(step.tool, step.args,
        allowed_tools, confirmed=confirmed)``:

        * If the step paused on the confirm-step (its result carries
          ``data["needs_confirmation"]`` and ``confirmed`` is ``False``), the run
          stops *before* the side-effecting work: ``ran=False``,
          ``needs_confirmation=True``, the paused step is reported, and NO later
          steps run.
        * If the step failed (``ok=False`` for any other reason — a tool error or a
          permission denial), the run stops and reports it (``ran=False``); no
          later steps run.
        * Otherwise the step is recorded as ``ok`` and the run continues.

        A protocol with all-successful steps returns ``ran=True``.
        """
        outcomes: list[StepOutcome] = []
        for step in protocol.steps:
            try:
                result = await self._registry.execute(
                    step.tool,
                    dict(step.args),
                    self._allowed_tools,
                    confirmed=confirmed,
                )
            except FridayPermissionError as exc:
                # An unpermitted / unregistered tool is denied before it runs.
                # Surface it honestly as a failed step and stop the run.
                outcomes.append(
                    StepOutcome(tool=step.tool, ok=False, error=str(exc))
                )
                return ProtocolResult(
                    protocol=protocol.name,
                    ran=False,
                    needs_confirmation=False,
                    steps=outcomes,
                )

            if result.data.get("needs_confirmation") and not confirmed:
                # The confirm-step paused this side-effecting step before it ran.
                # Report it and stop — later steps must NOT run.
                outcomes.append(
                    StepOutcome(
                        tool=step.tool,
                        ok=False,
                        error=None if result.error is None else result.error.code,
                        needs_confirmation=True,
                    )
                )
                return ProtocolResult(
                    protocol=protocol.name,
                    ran=False,
                    needs_confirmation=True,
                    steps=outcomes,
                )

            if not result.ok:
                # A genuine tool failure (bad args, tool error). Stop and report.
                outcomes.append(
                    StepOutcome(
                        tool=step.tool,
                        ok=False,
                        error=None if result.error is None else result.error.code,
                    )
                )
                return ProtocolResult(
                    protocol=protocol.name,
                    ran=False,
                    needs_confirmation=False,
                    steps=outcomes,
                )

            outcomes.append(StepOutcome(tool=step.tool, ok=True))

        return ProtocolResult(
            protocol=protocol.name,
            ran=True,
            needs_confirmation=False,
            steps=outcomes,
        )
