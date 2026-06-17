"""Unit tests for the alerting agent (Stage 3, build-spec section 4 / 9.7).

The :class:`~friday.agents.alerting.AlertingAgent` turns an alert request staged
in :class:`~friday.core.state.GraphState` into a single ``notify`` send while
collapsing duplicates. Its job is *deduplication + rate-limiting*: identical
alerts that arrive inside ``settings.alert_rate_limit_seconds`` of the last send
of that same alert must collapse to exactly ONE notification. A *distinct* alert
is always allowed through; an identical alert that arrives *after* the window has
elapsed is allowed through again.

Time is the crux, so it is **injected**: the agent reads "now" from a callable
clock handed in at construction, never from the wall clock. Tests drive that
clock by hand, so the windowing behaviour is fully deterministic and offline.
No network is touched — the ``notify`` tool's channel adapters are fakes — so no
``respx`` mocking is required.
"""

from __future__ import annotations

import pytest

from friday.agents.alerting import AlertingAgent
from friday.agents.base import Agent, AgentResult
from friday.config import Settings
from friday.core.state import GraphState, Mode
from friday.tools.notify import NotifyTool
from friday.tools.registry import ToolRegistry


class _Clock:
    """A hand-driven monotonic clock injected into the agent under test."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _registry() -> tuple[ToolRegistry, NotifyTool]:
    """A registry with a single FAKE :class:`NotifyTool`; return both."""
    notify = NotifyTool()
    registry = ToolRegistry()
    registry.register(notify)
    return registry, notify


def _state(
    *,
    channel: str = "slack",
    target: str = "#ops",
    subject: str = "Disk almost full",
    body: str = "Node n7 at 95% disk usage.",
    session_id: str = "alert-test",
) -> GraphState:
    """Build a graph state carrying an alert request in the scratchpad."""
    return GraphState(
        session_id=session_id,
        mode=Mode.ALERTING,
        user_input="raise an alert",
        scratchpad={
            "alert": {
                "channel": channel,
                "target": target,
                "subject": subject,
                "body": body,
            }
        },
    )


def _settings(*, window: float = 300.0, dedupe: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        alert_rate_limit_seconds=window,
        alert_dedupe=dedupe,
    )


# --------------------------------------------------------------------------- #
# Protocol + identity contract
# --------------------------------------------------------------------------- #
def test_alerting_agent_satisfies_agent_protocol() -> None:
    registry, _ = _registry()
    agent = AlertingAgent(registry, clock=_Clock(), settings=_settings())
    assert isinstance(agent, Agent)
    assert agent.name == "alerting"
    assert agent.allowed_tools == frozenset({"notify"})


# --------------------------------------------------------------------------- #
# The dedupe + rate-limit guarantee (the headline tests)
# --------------------------------------------------------------------------- #
async def test_dedupe_map_evicts_stale_entries() -> None:
    # Many distinct alerts are sent, then the clock advances past the window and
    # one more is sent: the dedupe map must not retain the now-useless old keys.
    registry, _ = _registry()
    clock = _Clock(start=1000.0)
    agent = AlertingAgent(registry, clock=clock, settings=_settings(window=300.0))

    for i in range(20):
        await agent.run(_state(subject=f"distinct alert {i}"))
    assert len(agent._last_sent) == 20  # all still inside the window

    clock.advance(301.0)  # the whole batch's window has now fully elapsed
    await agent.run(_state(subject="fresh alert"))

    # The 20 elapsed entries were evicted; only the fresh one remains.
    assert len(agent._last_sent) == 1


async def test_identical_alerts_within_window_collapse_to_one_send() -> None:
    # N identical alerts fired back-to-back inside the window -> exactly ONE
    # notify call. The clock does NOT advance between fires.
    registry, notify = _registry()
    clock = _Clock(start=1000.0)
    agent = AlertingAgent(registry, clock=clock, settings=_settings(window=300.0))

    results: list[AgentResult] = []
    for _ in range(5):
        results.append(await agent.run(_state()))

    # Exactly one message actually left via the (fake) notify tool.
    assert len(notify.sink) == 1
    # The first run sent; the rest were suppressed as duplicates.
    assert [r.tool_calls_made != [] for r in results] == [True, False, False, False, False]
    # The suppressed runs still return a coherent, non-empty AgentResult.
    for r in results:
        assert isinstance(r, AgentResult)
        assert r.output.strip() != ""


async def test_distinct_alert_sends_a_second_notification() -> None:
    # A DISTINCT alert (different subject/body/target) is never deduped against a
    # different alert: it gets its own send even within the same window.
    registry, notify = _registry()
    clock = _Clock(start=1000.0)
    agent = AlertingAgent(registry, clock=clock, settings=_settings(window=300.0))

    await agent.run(_state(subject="Disk almost full"))
    # Duplicate of the first -> suppressed.
    await agent.run(_state(subject="Disk almost full"))
    # A genuinely different alert -> a second send.
    await agent.run(_state(subject="CPU sustained 100%", body="Node n3 pegged."))

    assert len(notify.sink) == 2
    subjects = [m.subject for m in notify.sink]
    assert subjects == ["Disk almost full", "CPU sustained 100%"]


async def test_identical_alert_after_window_sends_again() -> None:
    # An identical alert that arrives AFTER the rate-limit window has elapsed is
    # allowed through again (the dedupe entry has expired).
    registry, notify = _registry()
    clock = _Clock(start=1000.0)
    agent = AlertingAgent(registry, clock=clock, settings=_settings(window=300.0))

    await agent.run(_state())  # send #1 at t=1000
    await agent.run(_state())  # duplicate at t=1000 -> suppressed
    assert len(notify.sink) == 1

    # Advance the injected clock PAST the window and fire the same alert again.
    clock.advance(300.0 + 1.0)
    await agent.run(_state())  # t=1301 -> window elapsed -> send #2
    assert len(notify.sink) == 2


async def test_alert_just_inside_window_is_still_suppressed() -> None:
    # A duplicate arriving just before the window closes is still a duplicate.
    registry, notify = _registry()
    clock = _Clock(start=1000.0)
    agent = AlertingAgent(registry, clock=clock, settings=_settings(window=300.0))

    await agent.run(_state())  # t=1000 send #1
    clock.advance(299.0)
    await agent.run(_state())  # t=1299 still inside 300s window -> suppressed

    assert len(notify.sink) == 1


# --------------------------------------------------------------------------- #
# Tool routing + result shape
# --------------------------------------------------------------------------- #
async def test_send_records_the_notify_tool_call() -> None:
    # The actually-sent run records the notify ToolCall it issued for audit.
    registry, notify = _registry()
    agent = AlertingAgent(registry, clock=_Clock(), settings=_settings())

    result = await agent.run(_state(channel="email", target="boss@example.com"))

    assert len(result.tool_calls_made) == 1
    call = result.tool_calls_made[0]
    assert call.name == "notify"
    assert call.arguments["channel"] == "email"
    assert call.arguments["target"] == "boss@example.com"
    # And the fake sink confirms exactly what would have been sent.
    assert len(notify.sink) == 1
    assert notify.sink[0].channel == "email"


async def test_confirm_step_is_satisfied_so_notify_actually_executes() -> None:
    # notify is side_effecting + non-idempotent, so without confirmation the
    # registry confirm-step would block it. The agent must pass confirmed=True so
    # the alert is genuinely dispatched (proven by the populated sink).
    registry, notify = _registry()
    agent = AlertingAgent(registry, clock=_Clock(), settings=_settings())

    result = await agent.run(_state())

    assert len(notify.sink) == 1
    assert result.tool_calls_made != []


async def test_dedupe_disabled_sends_every_time() -> None:
    # With dedupe turned off in settings, identical alerts each send (the agent
    # honours the configuration flag).
    registry, notify = _registry()
    agent = AlertingAgent(
        registry, clock=_Clock(), settings=_settings(dedupe=False)
    )

    await agent.run(_state())
    await agent.run(_state())
    await agent.run(_state())

    assert len(notify.sink) == 3


async def test_distinct_targets_are_independent_identities() -> None:
    # The same subject/body to two different targets are distinct alerts: each
    # gets its own send, and each is independently deduped thereafter.
    registry, notify = _registry()
    agent = AlertingAgent(registry, clock=_Clock(), settings=_settings())

    await agent.run(_state(target="#ops"))
    await agent.run(_state(target="#oncall"))
    await agent.run(_state(target="#ops"))  # duplicate of the first -> suppressed

    assert len(notify.sink) == 2
    assert {m.target for m in notify.sink} == {"#ops", "#oncall"}


async def test_missing_alert_in_scratchpad_is_a_graceful_no_send() -> None:
    # If the orchestrator staged no alert, the agent refuses cleanly rather than
    # crashing, and sends nothing.
    registry, notify = _registry()
    agent = AlertingAgent(registry, clock=_Clock(), settings=_settings())

    state = GraphState(
        session_id="empty",
        mode=Mode.ALERTING,
        user_input="alert",
        scratchpad={},
    )
    result = await agent.run(state)

    assert len(notify.sink) == 0
    assert result.tool_calls_made == []
    assert result.output.strip() != ""


def test_settings_default_when_not_injected() -> None:
    # The agent is constructible without an explicit Settings (it falls back to
    # get_settings), keeping it easy to wire.
    registry, _ = _registry()
    agent = AlertingAgent(registry, clock=_Clock())
    assert agent.name == "alerting"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
