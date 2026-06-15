"""Home / device-control tool with a FAKE actuator and a gated allow-list.

``HomeControlTool`` actuates a physical device *only* when both safety gates are
satisfied:

1. the ``enable_home`` feature flag is on, and
2. the requested ``device_id`` is in ``settings.device_allowlist``.

Either gate failing yields a typed refusal (``home_disabled`` /
``device_not_allowed``) — the action is never recorded. When both pass, the
action is appended to an in-memory ``sink`` (the fake actuator) and reported
back. The flag check is evaluated first so a globally-disabled home subsystem
refuses uniformly regardless of the allow-list.

The :class:`ActuatorPort` protocol describes the seam a real backend plugs into;
:class:`HomeAssistantActuator` is that real backend, present but **flagged off**
(every method raises ``NotImplementedError``) until Phase 4+.

The tool is ``side_effecting=True`` and ``idempotent=False`` so the registry
confirm-step (build-spec §12) gates it before execution.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from friday.config import get_settings
from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.home")


class HomeArgs(BaseModel):
    """Arguments for :class:`HomeControlTool`."""

    device_id: str
    action: str


class ActuatedAction(BaseModel):
    """A record of one action handed to the (fake) actuator."""

    device_id: str
    action: str


@runtime_checkable
class ActuatorPort(Protocol):
    """The seam a concrete device backend implements.

    ``actuate`` performs (or, for the fake, records) a single ``action`` against
    a ``device_id``. Implementations must assume the caller has already enforced
    the feature flag and allow-list gates.
    """

    async def actuate(self, device_id: str, action: str) -> None:
        """Perform the device action, or raise on an unsupported backend."""
        ...


class FakeActuator:
    """In-memory actuator: records actions instead of touching hardware."""

    def __init__(self) -> None:
        self.sink: list[ActuatedAction] = []

    async def actuate(self, device_id: str, action: str) -> None:
        self.sink.append(ActuatedAction(device_id=device_id, action=action))


class HomeAssistantActuator:
    """Real Home Assistant adapter — present but flagged off until Phase 4+."""

    async def actuate(self, device_id: str, action: str) -> None:
        raise NotImplementedError(
            "HomeAssistantActuator is not enabled until Phase 4+"
        )


class HomeControlTool:
    """Actuate an allow-listed device when the home subsystem is enabled.

    Safety order (part of the contract): the ``enable_home`` flag is checked
    first, then the ``device_allowlist`` membership; only then is the action
    recorded via the fake actuator and reported.
    """

    name = "home"
    description = "Control an allow-listed home device (on/off/toggle/etc.)."
    args_model = HomeArgs
    required_permission = "home"
    idempotent = False
    side_effecting = True

    def __init__(self) -> None:
        # Default to the fake actuator; a real backend can be injected later.
        self._actuator = FakeActuator()

    @property
    def sink(self) -> list[ActuatedAction]:
        """The fake actuator's record of actuated actions (for tests/audit)."""
        return self._actuator.sink

    async def __call__(self, args: Any) -> ToolResult:
        """Gate on flag + allow-list, then record the action via the actuator."""
        # ``args`` arrives validated from the registry; coerce defensively.
        if not isinstance(args, HomeArgs):
            args = HomeArgs.model_validate(args)

        settings = get_settings()

        if not settings.enable_home:
            logger.info(
                "home refused: subsystem disabled (device=%s action=%s)",
                args.device_id,
                args.action,
            )
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="home_disabled",
                    message="home control is disabled (enable_home is off)",
                    retriable=False,
                ),
            )

        if args.device_id not in settings.device_allowlist:
            logger.info(
                "home refused: device not allow-listed (device=%s action=%s)",
                args.device_id,
                args.action,
            )
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="device_not_allowed",
                    message=f"device {args.device_id!r} is not in the allow-list",
                    retriable=False,
                ),
            )

        await self._actuator.actuate(args.device_id, args.action)
        logger.info(
            "home actuated (fake) device=%s action=%s", args.device_id, args.action
        )
        return ToolResult(
            ok=True,
            data={"device_id": args.device_id, "action": args.action},
            error=None,
        )
