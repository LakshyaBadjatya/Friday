"""Notification tool with FAKE channel adapters.

``NotifyTool`` is the side-effecting, non-idempotent tool the alerting agent and
other callers use to reach an owner over email / Slack / a webhook. In this
phase the channel adapters are **fakes**: nothing leaves the process. Every
accepted message is appended to an in-memory ``sink`` exposed on the tool so
tests (and, later, an audit view) can assert exactly what *would* have been
sent. Real provider clients are deferred to a later phase.

Because the tool is ``side_effecting=True`` and ``idempotent=False`` the
registry's confirm-step (build-spec §12) gates it before execution; the tool
itself assumes confirmation already happened and simply records + reports.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel

from friday.tools.base import ToolResult

logger = logging.getLogger("friday.tools.notify")


class NotifyArgs(BaseModel):
    """Arguments for :class:`NotifyTool`.

    ``channel`` is constrained to the supported fake adapters; ``target`` is the
    channel-specific address (an email address, a Slack channel/handle, or a
    webhook URL). ``subject``/``body`` carry the message itself.
    """

    channel: Literal["email", "slack", "webhook"]
    target: str
    subject: str
    body: str


class SentMessage(BaseModel):
    """A record of one message handed to a (fake) channel adapter."""

    channel: Literal["email", "slack", "webhook"]
    target: str
    subject: str
    body: str


class NotifyTool:
    """Send a notification over a FAKE channel, recording it to an in-memory sink.

    No message is actually transmitted. Each successful call appends a
    :class:`SentMessage` to :attr:`sink` and returns
    ``ToolResult(ok=True, data={"sent": True, "channel": ..., "target": ...})``.
    """

    name = "notify"
    description = "Send a notification over email, Slack, or a webhook (owner-facing)."
    args_model = NotifyArgs
    required_permission = "notify"
    idempotent = False
    side_effecting = True

    def __init__(self) -> None:
        # In-memory sink of everything that *would* have been sent. Exposed for
        # tests and audit; per-instance so each tool/agent gets an isolated log.
        self.sink: list[SentMessage] = []

    async def __call__(self, args: Any) -> ToolResult:
        """Record the message to the fake sink and report it as sent."""
        # ``args`` arrives validated from the registry; coerce defensively so the
        # tool is also safe to call directly with a raw mapping.
        if not isinstance(args, NotifyArgs):
            args = NotifyArgs.model_validate(args)

        message = SentMessage(
            channel=args.channel,
            target=args.target,
            subject=args.subject,
            body=args.body,
        )
        self.sink.append(message)
        logger.info(
            "notify (fake) channel=%s target=%s subject=%r",
            args.channel,
            args.target,
            args.subject,
        )
        return ToolResult(
            ok=True,
            data={"sent": True, "channel": args.channel, "target": args.target},
            error=None,
        )
