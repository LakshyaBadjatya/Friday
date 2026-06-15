"""Example FRIDAY plugin: a deterministic ``dice_roll`` tool.

Drop a file like this into your ``FRIDAY_PLUGINS_DIR`` (default ``plugins/``) and
FRIDAY will discover it at startup and register the tools its module-level
``get_tools()`` returns into the shared tool registry — so they go through the
same permission gating + confirm-step as the built-in tools.

This example is intentionally trivial and fully deterministic: ``dice_roll``
takes an explicit ``seed`` and derives the roll from it, so there is **no
randomness at import time** and no wall-clock dependency. The same ``(seed,
sides, count)`` always yields the same roll, which keeps the build offline and
reproducible.

See ``README.md`` in this directory for the plugin convention and the
trusted-code caveat.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from friday.tools.base import Tool, ToolError, ToolResult


class DiceRollArgs(BaseModel):
    """Arguments for :class:`DiceRollTool`.

    ``seed`` makes the roll deterministic (required — no implicit randomness).
    ``sides`` is the die size and ``count`` how many dice to roll.
    """

    seed: int
    sides: int = Field(default=6, ge=2, le=1000)
    count: int = Field(default=1, ge=1, le=100)


class DiceRollTool:
    """Roll ``count`` dice of ``sides`` faces deterministically from a ``seed``.

    Read-only and idempotent: it touches nothing external, so it never trips the
    confirm-step. The randomness is seeded entirely by the explicit ``seed``
    argument, so a given ``(seed, sides, count)`` is reproducible.
    """

    name = "dice_roll"
    description = "Roll dice deterministically from an explicit seed."
    args_model = DiceRollArgs
    required_permission = "dice_roll"
    idempotent = True
    side_effecting = False

    async def __call__(self, args: Any) -> ToolResult:
        """Roll the dice for the validated ``args`` and return the rolls + total."""
        if not isinstance(args, DiceRollArgs):
            args = DiceRollArgs.model_validate(args)
        if args.sides < 2:  # pragma: no cover - guarded by args_model ge=2
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="bad_args", message="a die needs at least 2 sides"
                ),
            )
        rng = random.Random(args.seed)
        rolls = [rng.randint(1, args.sides) for _ in range(args.count)]
        return ToolResult(
            ok=True,
            data={
                "seed": args.seed,
                "sides": args.sides,
                "rolls": rolls,
                "total": sum(rolls),
            },
        )


def get_tools() -> list[Tool]:
    """The plugin entry point: return the tools this plugin contributes.

    FRIDAY's plugin loader calls this module-level function and registers each
    returned :class:`~friday.tools.base.Tool` instance into the shared registry.
    """
    return [DiceRollTool()]
