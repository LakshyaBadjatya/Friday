"""The device agent: confirm-gated, allow-listed home/device control.

:class:`DeviceAgent` is the FRIDAY specialist for "turn on / off / toggle the
lights / thermostat / plug" turns. It implements the
:class:`~friday.agents.base.Agent` protocol (``name="device"``,
``allowed_tools={"home"}``) and does exactly one thing: route a single device
action through the ``home`` tool via the injected
:class:`~friday.tools.registry.ToolRegistry`.

It owns **none** of the safety policy itself — that is deliberate. The two
guarantees that make device control safe live downstream and the agent merely
respects them:

* **The registry confirm-step (build-spec section 12).** The ``home`` tool is
  ``side_effecting`` and not ``idempotent``, so the registry refuses to execute
  it unless ``confirmed=True``. The agent threads ``state.confirmed`` straight
  into ``registry.execute(...)`` — it never sets it itself — so an unconfirmed
  action comes back as ``confirmation_required`` and nothing actuates.
* **The home tool's flag + allow-list gates.** The ``home`` tool refuses with
  ``home_disabled`` (the ``enable_home`` flag is off) or ``device_not_allowed``
  (the ``device_id`` is not on ``settings.device_allowlist``) before touching
  any actuator.

The agent's job on a refusal is to **surface it honestly** — it never fabricates
a success the tool did not grant. A refusal yields a low-confidence
:class:`~friday.agents.base.AgentResult` that names the cause; only a real,
``ok=True`` tool result yields a confident success.

There is no LLM and no network in this path — only the registry — so the module
keeps ``friday.agents`` clean of provider SDKs (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import logging
import uuid

from friday.agents.base import AgentResult
from friday.core.state import GraphState
from friday.errors import PermissionError
from friday.providers.llm import ToolCall
from friday.tools.base import ToolResult
from friday.tools.registry import ToolRegistry

logger = logging.getLogger("friday.agents.device")

# The single tool this agent is permitted to reach.
_ALLOWED_TOOLS: frozenset[str] = frozenset({"home"})


class DeviceAgent:
    """Route a staged device action through the confirm-gated ``home`` tool.

    The action is read from ``state.scratchpad['device']`` as a
    ``{"device_id": ..., "action": ...}`` mapping (the orchestrator stages it
    there; state round-trips through JSON, so it arrives as a plain dict). The
    agent validates that shape, then dispatches via the registry — honouring the
    allow-list, the ``enable_home`` flag, and the confirm-step — and reports the
    real outcome without embellishment.

    Args:
        registry: The tool registry the agent dispatches ``home`` through.
    """

    name = "device"
    allowed_tools = _ALLOWED_TOOLS

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # -- action extraction -------------------------------------------------- #
    @staticmethod
    def _action_from_state(state: GraphState) -> tuple[str, str] | None:
        """Pull ``(device_id, action)`` staged in ``scratchpad['device']``.

        Returns ``None`` when nothing usable was staged (missing key, wrong
        type, or blank fields) so the caller can refuse rather than dispatch a
        malformed call.
        """
        raw = state.scratchpad.get("device")
        if not isinstance(raw, dict):
            return None
        device_id = raw.get("device_id")
        action = raw.get("action")
        if not isinstance(device_id, str) or not isinstance(action, str):
            return None
        if not device_id.strip() or not action.strip():
            return None
        return device_id, action

    # -- refusal helper ----------------------------------------------------- #
    @staticmethod
    def _refusal(message: str, *, tool_calls: list[ToolCall]) -> AgentResult:
        """A low-confidence result that surfaces a refusal without fabrication."""
        return AgentResult(
            output=message,
            tool_calls_made=tool_calls,
            confidence=0.3,
        )

    # -- agent entrypoint --------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Dispatch the staged device action through the ``home`` tool.

        Threads ``state.confirmed`` into the registry confirm-step so a
        side-effecting action only executes once explicitly confirmed. Returns a
        confident success only on a real ``ok=True`` tool result; every refusal
        (no action staged, permission denial, confirmation required, or a tool
        ``ok=False``) is surfaced as a low-confidence, honest message.
        """
        extracted = self._action_from_state(state)
        if extracted is None:
            logger.info("device: no actionable request staged in scratchpad")
            return self._refusal(
                "I don't have a clear device and action to act on, so I won't "
                "guess. Tell me which device to control and what to do.",
                tool_calls=[],
            )

        device_id, action = extracted
        raw_args: dict[str, object] = {"device_id": device_id, "action": action}
        call = ToolCall(
            id=f"call_{uuid.uuid4().hex}", name="home", arguments=raw_args
        )

        try:
            result: ToolResult = await self._registry.execute(
                "home",
                raw_args,
                allowed_tools=self.allowed_tools,
                confirmed=state.confirmed,
            )
        except PermissionError as exc:  # pragma: no cover - defensive
            logger.warning("device denied home: %s", exc)
            return self._refusal(
                f"I'm not permitted to control {device_id!r} right now.",
                tool_calls=[call],
            )

        if not result.ok:
            code = result.error.code if result.error is not None else "error"
            detail = (
                result.error.message if result.error is not None else "unknown error"
            )
            logger.info(
                "device refused device=%s action=%s code=%s",
                device_id,
                action,
                code,
            )
            return self._refusal(
                f"I couldn't {action} {device_id!r}: {code} ({detail}).",
                tool_calls=[call],
            )

        logger.info("device actuated device=%s action=%s", device_id, action)
        return AgentResult(
            output=f"Done — {action} on {device_id!r}.",
            tool_calls_made=[call],
            confidence=1.0,
        )
