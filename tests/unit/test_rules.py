# © Lakshya Badjatya — Author
"""Unit tests for the local IFTTT-style rules engine."""

from __future__ import annotations

from friday.proactive.rules import Action, Condition, Rule, RulesEngine


def _rule(**kw: object) -> Rule:
    kw.setdefault("action", Action(name="notify"))
    return Rule(**kw)  # type: ignore[arg-type]


def test_fires_on_matching_trigger_no_condition() -> None:
    engine = RulesEngine([_rule(name="r1", trigger="cpu_high")])
    fired = engine.evaluate("cpu_high", {})
    assert [f.rule for f in fired] == ["r1"]
    assert engine.evaluate("other", {}) == []


def test_condition_gates_firing() -> None:
    rule = _rule(
        name="hot",
        trigger="metric",
        condition=Condition(field="cpu", op="gt", value=90),
    )
    engine = RulesEngine([rule])
    assert [f.rule for f in engine.evaluate("metric", {"cpu": 95})] == ["hot"]
    assert engine.evaluate("metric", {"cpu": 50}) == []


def test_disabled_rule_never_fires() -> None:
    engine = RulesEngine([_rule(name="r", trigger="e", enabled=False)])
    assert engine.evaluate("e", {}) == []


def test_operators() -> None:
    def fires(op: str, value: object, actual: object) -> bool:
        rule = _rule(name="x", trigger="e", condition=Condition(field="f", op=op, value=value))  # type: ignore[arg-type]
        return bool(RulesEngine([rule]).evaluate("e", {"f": actual}))

    assert fires("eq", 1, 1) and not fires("eq", 1, 2)
    assert fires("ne", 1, 2)
    assert fires("gte", 90, 90) and fires("lte", 90, 90)
    assert fires("contains", "err", "an error") and not fires("contains", "x", "abc")
    assert fires("contains", "a", ["a", "b"])  # membership for sequences
    assert fires("exists", None, "anything")


def test_missing_field_does_not_fire_for_value_ops() -> None:
    rule = _rule(name="x", trigger="e", condition=Condition(field="absent", op="eq", value=1))
    assert RulesEngine([rule]).evaluate("e", {"present": 1}) == []


def test_noncomparable_types_do_not_order() -> None:
    rule = _rule(name="x", trigger="e", condition=Condition(field="f", op="gt", value=5))
    # A non-numeric actual can't be ordered -> no fire (no crash).
    assert RulesEngine([rule]).evaluate("e", {"f": "string"}) == []
    # bool is excluded from ordering too.
    rule_b = _rule(name="y", trigger="e", condition=Condition(field="f", op="gt", value=0))
    assert RulesEngine([rule_b]).evaluate("e", {"f": True}) == []


def test_action_payload_passthrough() -> None:
    engine = RulesEngine(
        [_rule(name="r", trigger="e", action=Action(name="lockdown", args={"scope": "owner"}))]
    )
    fired = engine.evaluate("e", {})
    assert fired[0].action.name == "lockdown"
    assert fired[0].action.args == {"scope": "owner"}
