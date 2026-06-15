"""Unit tests for the write-consent policy (Phase 4 Stage 2, build-spec §10).

An agent proposes memory writes by appending :class:`MemoryWrite` records to its
:class:`AgentResult.memory_writes`. The orchestrator applies a consent policy
*after* the agent runs:

* **Non-sensitive** writes are auto-committed to the long-term store when
  ``settings.memory_autowrite`` is true (and never when it is false).
* **Sensitive** writes are NEVER auto-persisted. The orchestrator surfaces a
  persona confirm prompt and commits them only on a confirming follow-up
  (``state.confirmed=True``).

These tests use a real in-process :class:`SQLiteLongTermStore` (``":memory:"``)
and a scripted :class:`FakeLLM`, so they are fully offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.agents.base import AgentResult
from friday.core.orchestrator import MemoryWrite, Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Message, ToolSpec, Usage
from friday.tools.registry import ToolRegistry

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


class _ScriptedAgent:
    """A minimal agent that returns a fixed :class:`AgentResult`.

    Registered under one of the dispatch modes so the orchestrator runs it and
    then applies the consent policy to its proposed ``memory_writes``.
    """

    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, name: str, result: AgentResult) -> None:
        self.name = name
        self._result = result

    async def run(self, state: GraphState) -> AgentResult:
        return self._result


class _EchoLLM(FakeLLM):
    """A FakeLLM that echoes the last user message so persona-wrap never starves."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user" and m.content),
            "ok",
        )
        return LLMResponse(text=last_user, tool_calls=[], usage=Usage())


def _orchestrator(
    long_term: SQLiteLongTermStore,
    agent: _ScriptedAgent,
    *,
    autowrite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> Orchestrator:
    from friday import config
    from friday.agents.base import AgentRegistry

    settings = config.Settings(memory_autowrite=autowrite)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    # The orchestrator + router both read get_settings from their own modules.
    import friday.core.orchestrator as orch_mod
    import friday.core.router as router_mod

    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(router_mod, "get_settings", lambda: settings)

    agents = AgentRegistry()
    agents.register(agent)
    return Orchestrator(
        llm=_EchoLLM(responses=[]),
        registry=ToolRegistry(),
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        agents=agents,
        long_term=long_term,
    )


def _automation_state(text: str, *, confirmed: bool = False) -> GraphState:
    # "automate" routes to AUTOMATION, which is NOT confirm-gated at the
    # orchestrator level, so the agent runs and the consent policy applies.
    return GraphState(
        session_id="consent-test",
        mode=Mode.AUTOMATION,
        user_input=text,
        confirmed=confirmed,
    )


async def test_non_sensitive_write_auto_commits_when_autowrite_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteLongTermStore(":memory:")
    agent = _ScriptedAgent(
        "automation",
        AgentResult(
            output="Scheduled the backup job.",
            memory_writes=[
                MemoryWrite(
                    text="The owner's backup job runs nightly at 2am.",
                    source_id="automation-backup",
                    sensitive=False,
                )
            ],
        ),
    )
    orch = _orchestrator(store, agent, autowrite=True, monkeypatch=monkeypatch)

    await orch.handle(_automation_state("automate the nightly backup"))

    # The non-sensitive write was auto-committed: a query finds it.
    facts = store.query_facts("backup job runs nightly")
    assert len(facts) == 1
    assert facts[0].source_id == "automation-backup"
    assert facts[0].sensitive is False


async def test_non_sensitive_write_not_committed_when_autowrite_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteLongTermStore(":memory:")
    agent = _ScriptedAgent(
        "automation",
        AgentResult(
            output="Scheduled the backup job.",
            memory_writes=[
                MemoryWrite(
                    text="The owner's backup job runs nightly at 2am.",
                    source_id="automation-backup",
                    sensitive=False,
                )
            ],
        ),
    )
    orch = _orchestrator(store, agent, autowrite=False, monkeypatch=monkeypatch)

    await orch.handle(_automation_state("automate the nightly backup"))

    # Autowrite off -> nothing persisted, even for a non-sensitive write.
    assert store.query_facts("backup") == []


async def test_sensitive_write_not_committed_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteLongTermStore(":memory:")
    agent = _ScriptedAgent(
        "automation",
        AgentResult(
            output="Noted.",
            memory_writes=[
                MemoryWrite(
                    text="The owner's home alarm code is 4827.",
                    source_id="automation-alarm",
                    sensitive=True,
                )
            ],
        ),
    )
    orch = _orchestrator(store, agent, autowrite=True, monkeypatch=monkeypatch)

    state = await orch.handle(_automation_state("automate arming the alarm"))

    # Sensitive data is NEVER auto-persisted, even with autowrite on.
    assert store.query_facts("alarm code") == []
    # And the owner is asked to confirm before it is stored.
    assert state.response is not None
    lowered = state.response.lower()
    assert "confirm" in lowered or "remember" in lowered or "store" in lowered


async def test_sensitive_write_committed_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteLongTermStore(":memory:")
    agent = _ScriptedAgent(
        "automation",
        AgentResult(
            output="Noted.",
            memory_writes=[
                MemoryWrite(
                    text="The owner's home alarm code is 4827.",
                    source_id="automation-alarm",
                    sensitive=True,
                )
            ],
        ),
    )
    orch = _orchestrator(store, agent, autowrite=True, monkeypatch=monkeypatch)

    # First turn proposes the sensitive write (pending, not stored).
    await orch.handle(_automation_state("automate arming the alarm"))
    assert store.query_facts("alarm code") == []

    # A confirming follow-up turn commits the pending sensitive write.
    confirmed = await orch.handle(
        _automation_state("yes, go ahead and remember that", confirmed=True)
    )

    facts = store.query_facts("alarm code")
    assert len(facts) == 1
    assert facts[0].source_id == "automation-alarm"
    assert facts[0].sensitive is True
    assert confirmed.response is not None
