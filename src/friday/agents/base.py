"""Agent boundary: the :class:`Agent` protocol and :class:`AgentResult` payload.

An *agent* is a specialized worker the orchestrator dispatches a turn to once a
:class:`~friday.core.state.RouteDecision` selects it. Each agent declares the
tools it is permitted to use (``allowed_tools``) — the registry enforces this
allow-list, so an agent can never reach a tool it did not declare.

Only a minimal Research path is exercised this phase; the remaining agents are
Phase 2. This module therefore defines the *contract* (protocol + result model)
without committing to any concrete agent beyond what the orchestrator needs.

``AgentResult`` is the normalized return of an agent run:

* ``output`` — the agent's draft answer (pre-persona synthesis).
* ``tool_calls_made`` — the tool calls the agent actually issued, for audit.
* ``memory_writes`` — opaque records the orchestrator may persist to memory.
* ``confidence`` — the agent's own calibrated confidence in ``output``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.core.state import GraphState
from friday.providers.llm import ToolCall


class AgentResult(BaseModel):
    """The normalized result of a single :class:`Agent` run."""

    output: str
    tool_calls_made: list[ToolCall] = Field(default_factory=list)
    memory_writes: list[Any] = Field(default_factory=list)
    confidence: float = 1.0


@runtime_checkable
class Agent(Protocol):
    """Structural contract every FRIDAY agent implements.

    ``name`` identifies the agent to the router/orchestrator. ``allowed_tools``
    is the frozen set of tool names the agent may invoke — the registry rejects
    anything outside it. ``run`` performs the work against the current
    :class:`GraphState` and returns an :class:`AgentResult`.
    """

    name: str
    allowed_tools: frozenset[str]

    async def run(self, state: GraphState) -> AgentResult:
        """Execute the agent for one turn and return its result."""
        ...


class AgentRegistry:
    """An in-process registry mapping an agent's ``name`` to its instance.

    The orchestrator dispatches a routed turn by looking up the agent for the
    decided :class:`~friday.core.state.Mode` here. It is the agent-level analogue
    of :class:`~friday.tools.registry.ToolRegistry`: ``app.py`` constructs each
    agent with its dependencies (the tool registry, the vector store, the LLM
    provider) and registers it, then injects the populated registry into the
    orchestrator. Lookups are by the agent's own ``name``.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register ``agent`` under its ``name``, replacing any prior entry."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Return the registered agent named ``name`` or raise :class:`KeyError`."""
        return self._agents[name]

    def __contains__(self, name: object) -> bool:
        """Whether an agent is registered under ``name``."""
        return name in self._agents
