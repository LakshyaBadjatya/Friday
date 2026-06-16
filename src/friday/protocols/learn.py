# © Lakshya Badjatya — Author
"""Skill / macro learning: turn a recent tool-call sequence into a named protocol.

"Remember how I did that as a protocol." After the owner walks FRIDAY through a
sequence of actions, every tool the registry ran left a :class:`ToolCallAudit`
row in the audit log. This module folds such a sequence into a draft
:class:`~friday.protocols.store.Protocol` — an ordered list of
:class:`~friday.protocols.store.ProtocolStep` (tool + args) — that the existing
protocol store/runner can replay on a single trigger phrase.

Two safety properties by construction:

* The learned protocol is created **disabled** (``enabled=False``) so it never
  starts firing until the owner has reviewed and turned it on.
* Steps carry the audit log's **redacted** args, so a secret passed as a tool
  argument is never baked verbatim into a stored, replayable macro.
  :func:`has_redacted_args` lets the caller warn the owner that any redacted
  value must be filled in before the macro will replay correctly.

This is a pure transform: it imports no LLM SDK, reads no configuration, and
performs no I/O — persistence is the caller's job (via the protocol store, which
assigns the real id; the returned draft carries the placeholder ``id=0``).
"""

from __future__ import annotations

from collections.abc import Iterable

from friday.logging import REDACTED
from friday.observability.audit import ToolCallAudit
from friday.protocols.store import Protocol, ProtocolStep


def learn_protocol(
    name: str,
    trigger_phrase: str,
    calls: Iterable[ToolCallAudit],
    *,
    only_successful: bool = True,
    include_tools: Iterable[str] | None = None,
) -> Protocol:
    """Fold an audited tool-call sequence into a draft (disabled) :class:`Protocol`.

    Args:
        name: The protocol's name (must be non-blank).
        trigger_phrase: The phrase that will fire it once enabled (non-blank).
        calls: The audited tool calls to learn from, in execution order.
        only_successful: Skip calls that did not succeed (``ok is False``) so a
            learned macro replays only steps that actually worked. Default true.
        include_tools: If given, keep only calls to these tool names (drop
            incidental calls); ``None`` keeps every tool.

    Returns:
        A :class:`Protocol` with ``enabled=False`` and ``id=0`` (the store assigns
        the real id on save), its steps in the calls' original order.

    Raises:
        ValueError: if ``name`` / ``trigger_phrase`` is blank, or no calls remain
            after filtering (nothing to learn).
    """
    clean_name = name.strip()
    clean_trigger = trigger_phrase.strip()
    if not clean_name:
        raise ValueError("protocol name must not be blank")
    if not clean_trigger:
        raise ValueError("trigger phrase must not be blank")

    allowed = set(include_tools) if include_tools is not None else None
    steps: list[ProtocolStep] = []
    for call in calls:
        if only_successful and not call.ok:
            continue
        if allowed is not None and call.tool not in allowed:
            continue
        steps.append(ProtocolStep(tool=call.tool, args=dict(call.args_redacted)))

    if not steps:
        raise ValueError("no tool calls to learn from (after filtering)")

    return Protocol(
        id=0,
        name=clean_name,
        trigger_phrase=clean_trigger,
        steps=steps,
        enabled=False,
    )


def has_redacted_args(protocol: Protocol) -> bool:
    """Whether any step carries a redacted secret value that must be filled in.

    A learned step's args come from the audit log, where secret-keyed values are
    replaced with the :data:`~friday.logging.REDACTED` sentinel. If any remain,
    the macro cannot replay verbatim until the owner supplies the real value, so
    the caller should surface a warning before enabling it.
    """
    return any(REDACTED in step.args.values() for step in protocol.steps)
