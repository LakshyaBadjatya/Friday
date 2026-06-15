"""Mode node functions over :class:`GraphState` (Task 1.9).

These are thin wrappers binding an :class:`~friday.core.orchestrator.Orchestrator`
to the four mode nodes the graph wires together: ROUTING, CONVERSATION, RESEARCH,
and CLARIFY. Each returns the (mutated) :class:`GraphState` so it composes cleanly
whether driven by LangGraph or by the fallback state machine — both consume the
same node contract.

The functions delegate *all* behaviour to the orchestrator's ``node_*`` methods;
they exist as a stable, framework-agnostic seam so ``graph.py`` can register node
callables without reaching into orchestrator internals.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState


class ModeNode(Protocol):
    """A mode node: async, takes the live state, returns the advanced state.

    Declared as a :class:`~typing.Protocol` (rather than a ``Callable`` alias)
    so it matches LangGraph's node-input inference in ``StateGraph.add_node``
    while still expressing the exact ``(GraphState) -> Awaitable[GraphState]``
    shape both engines rely on.
    """

    def __call__(self, state: GraphState) -> Awaitable[GraphState]:
        """Advance ``state`` by one mode step."""
        ...


def routing_node(orchestrator: Orchestrator) -> ModeNode:
    """Build the ROUTING node bound to ``orchestrator``."""

    async def _node(state: GraphState) -> GraphState:
        return await orchestrator.node_routing(state)

    return _node


def conversation_node(orchestrator: Orchestrator) -> ModeNode:
    """Build the CONVERSATION node bound to ``orchestrator``."""

    async def _node(state: GraphState) -> GraphState:
        return await orchestrator.node_conversation(state)

    return _node


def research_node(orchestrator: Orchestrator) -> ModeNode:
    """Build the RESEARCH node bound to ``orchestrator``."""

    async def _node(state: GraphState) -> GraphState:
        return await orchestrator.node_research(state)

    return _node


def clarify_node(orchestrator: Orchestrator) -> ModeNode:
    """Build the CLARIFY node bound to ``orchestrator``."""

    async def _node(state: GraphState) -> GraphState:
        return await orchestrator.node_clarify(state)

    return _node
