"""Ask-user tool: let the agent pause and request input from the owner.

:class:`AskUserTool` is the agent's structured way to say "I need more from you
before I can continue." It performs no work and reaches nothing external: it
simply echoes a question (and optional multiple-choice ``options``) back inside a
:class:`~friday.tools.base.ToolResult` flagged with ``needs_input=True``. The
orchestrator/agent loop recognizes that flag and pauses the turn, surfacing the
question to the user, then resumes with their answer.

This mirrors the registry's ``needs_confirmation`` convention (build-spec §12) —
a result the loop interprets rather than a hard error — but for *input gathering*
rather than *side-effect confirmation*.

The tool is pure and dependency-free: it takes no collaborators, imports neither
:mod:`friday.config` nor :mod:`friday.app`, and is read-only
(``side_effecting=False``, ``idempotent=True``). Asking the same question twice
is harmless and produces the same payload.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from friday.tools.base import ToolResult

logger = logging.getLogger("friday.tools.ask_user")


class AskUserArgs(BaseModel):
    """Arguments for :class:`AskUserTool`.

    ``question`` is the prompt shown to the owner. ``options`` is an optional list
    of suggested answers for a multiple-choice question; omit it for a free-text
    prompt. An empty ``options`` list is treated the same as ``None`` (no choices).
    """

    question: str = Field(min_length=1)
    options: list[str] | None = None


class AskUserTool:
    """Pause the turn to ask the owner a question (optionally multiple-choice).

    Pure and stateless: it takes no constructor dependencies. ``__call__`` returns
    ``ToolResult(ok=True, data={"needs_input": True, "question": ..., "options":
    ...})`` so the agent loop can detect the pause signal and route the question
    to the user instead of treating the result as a final answer.
    """

    name = "ask_user"
    description = (
        "Ask the user a clarifying question before continuing. Optionally provide "
        "a list of suggested options for a multiple-choice answer."
    )
    args_model = AskUserArgs
    required_permission = "ask_user"
    idempotent = True
    side_effecting = False

    async def __call__(self, args: Any) -> ToolResult:
        """Return the pause signal payload for ``args.question`` / ``args.options``."""
        if not isinstance(args, AskUserArgs):
            args = AskUserArgs.model_validate(args)
        # Normalize an empty options list to ``None`` so the loop sees a single,
        # unambiguous "free-text vs choices" signal.
        options = args.options if args.options else None
        data: dict[str, Any] = {
            "needs_input": True,
            "question": args.question,
            "options": options,
        }
        logger.info(
            "ask_user question=%r options=%d",
            args.question,
            0 if options is None else len(options),
        )
        return ToolResult(ok=True, data=data, error=None)
