# © Lakshya Badjatya — Author
"""Unit tests for skill/macro learning (audit tool-calls -> draft protocol)."""

from __future__ import annotations

import pytest

from friday.logging import REDACTED
from friday.observability.audit import ToolCallAudit
from friday.protocols.learn import has_redacted_args, learn_protocol


def _call(tool: str, *, ok: bool = True, args: dict[str, object] | None = None) -> ToolCallAudit:
    return ToolCallAudit(
        correlation_id="c1", tool=tool, args_redacted=args or {}, ok=ok, ts=0.0
    )


def test_learn_preserves_order_and_is_disabled() -> None:
    calls = [_call("notify", args={"text": "night"}), _call("home", args={"device": "lights"})]
    proto = learn_protocol("Goodnight", "goodnight", calls)
    assert [s.tool for s in proto.steps] == ["notify", "home"]
    assert proto.steps[0].args == {"text": "night"}
    assert proto.enabled is False  # created disabled for review
    assert proto.id == 0  # placeholder; store assigns the real id


def test_only_successful_filters_failed_calls() -> None:
    calls = [_call("notify"), _call("home", ok=False), _call("create_reminder")]
    proto = learn_protocol("p", "trig", calls)
    assert [s.tool for s in proto.steps] == ["notify", "create_reminder"]
    # ...unless explicitly told to keep failures
    proto2 = learn_protocol("p", "trig", calls, only_successful=False)
    assert [s.tool for s in proto2.steps] == ["notify", "home", "create_reminder"]


def test_include_tools_filters() -> None:
    calls = [_call("notify"), _call("web_search"), _call("home")]
    proto = learn_protocol("p", "trig", calls, include_tools={"notify", "home"})
    assert [s.tool for s in proto.steps] == ["notify", "home"]


def test_blank_name_or_trigger_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        learn_protocol("  ", "trig", [_call("notify")])
    with pytest.raises(ValueError, match="trigger"):
        learn_protocol("p", "  ", [_call("notify")])


def test_no_usable_calls_rejected() -> None:
    with pytest.raises(ValueError, match="no tool calls"):
        learn_protocol("p", "trig", [_call("home", ok=False)])  # all filtered out


def test_has_redacted_args_flags_secrets() -> None:
    plain = learn_protocol("p", "trig", [_call("notify", args={"text": "hi"})])
    assert has_redacted_args(plain) is False
    secret = learn_protocol("p", "trig", [_call("notify", args={"token": REDACTED})])
    assert has_redacted_args(secret) is True
