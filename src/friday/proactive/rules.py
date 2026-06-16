# © Lakshya Badjatya — Author
"""A local IFTTT-style rules engine: trigger -> condition -> action.

The proactive spine reacts to events ("a metric breached", "a reminder came
due", "presence changed") by firing owner-defined rules — no external automation
service needed. A :class:`Rule` names the event that triggers it, an optional
:class:`Condition` over the event payload, and the :class:`Action` to fire when
both match. :class:`RulesEngine.evaluate` returns the actions that fired for an
event; *executing* them (through the broker, so every side effect is gated and
audited) is the caller's job.

This module is the pure decision core: rules and payloads are plain data, the
condition language is a small fixed set of comparisons (no arbitrary code), and
evaluation is deterministic. It imports no LLM SDK, reads no configuration, and
performs no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

#: The comparison operators a rule condition may use.
ConditionOp = Literal["eq", "ne", "gt", "lt", "gte", "lte", "contains", "exists"]


class Condition(BaseModel):
    """A single comparison against a field of the event payload.

    Attributes:
        field: The payload key to read.
        op: The comparison to apply.
        value: The right-hand operand (ignored for ``exists``).
    """

    field: str
    op: ConditionOp
    value: Any = None

    def matches(self, payload: dict[str, Any]) -> bool:
        """Whether this condition holds for ``payload`` (False if it can't apply)."""
        if self.op == "exists":
            return self.field in payload
        if self.field not in payload:
            return False
        actual = payload[self.field]
        if self.op == "eq":
            return bool(actual == self.value)
        if self.op == "ne":
            return bool(actual != self.value)
        if self.op in ("gt", "lt", "gte", "lte"):
            if not _orderable(actual) or not _orderable(self.value):
                return False
            if self.op == "gt":
                return bool(actual > self.value)
            if self.op == "lt":
                return bool(actual < self.value)
            if self.op == "gte":
                return bool(actual >= self.value)
            return bool(actual <= self.value)
        # contains: substring for strings, membership for sequences.
        if isinstance(actual, str):
            return isinstance(self.value, str) and self.value in actual
        if isinstance(actual, (list, tuple, set)):
            return self.value in actual
        return False


def _orderable(value: Any) -> bool:
    """Whether ``value`` is a real number (bool excluded) usable in </> compares."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Action(BaseModel):
    """The action a rule fires: a named action plus its arguments."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    """A trigger/condition/action rule.

    Attributes:
        name: A human label for the rule.
        trigger: The event name that arms this rule.
        action: The action to fire when the rule matches.
        condition: Optional gate on the event payload; ``None`` always passes.
        enabled: Whether the rule participates in evaluation.
    """

    name: str
    trigger: str
    action: Action
    condition: Condition | None = None
    enabled: bool = True


class FiredRule(BaseModel):
    """A rule that matched an event, with the action to execute."""

    rule: str
    action: Action


class RulesEngine:
    """Evaluates a set of rules against incoming events."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules: list[Rule] = list(rules)

    def evaluate(self, event: str, payload: dict[str, Any] | None = None) -> list[FiredRule]:
        """Return the actions of every enabled rule that fires for ``event``.

        A rule fires when it is enabled, its ``trigger`` equals ``event``, and its
        condition (if any) matches ``payload``. Rules are evaluated in their
        declared order, so the returned actions are deterministic.
        """
        data = payload or {}
        fired: list[FiredRule] = []
        for rule in self._rules:
            if not rule.enabled or rule.trigger != event:
                continue
            if rule.condition is not None and not rule.condition.matches(data):
                continue
            fired.append(FiredRule(rule=rule.name, action=rule.action))
        return fired
