"""Mode-loop assembly: a LangGraph ``StateGraph`` (with a state-machine fallback).

This wires the four mode nodes from :mod:`friday.core.modes` into the turn flow:

    START -> routing -> (conditional on mode) -> {conversation | research | clarify} -> END

The ROUTING node classifies the turn and persists ``mode`` into the state; a
conditional edge then dispatches to exactly one mode node, which produces the
``response`` and ends the turn.

**Portability contract.** Both the LangGraph build and the fallback expose the
identical surface — ``async def invoke(state: GraphState) -> GraphState`` — so
callers (the app, tests) never care which engine is underneath. LangGraph 1.x is
the primary engine; if it ever fails to import or compile (e.g. a Python 3.14
incompatibility) :func:`build_graph` transparently falls back to
:class:`_StateMachine`, a minimal hand-rolled async equivalent, and logs the
decision. On the pinned toolchain (langgraph 1.2.5, Python 3.14.4) the LangGraph
path is exercised; the fallback exists purely as a resilience seam.
"""

from __future__ import annotations

import logging

from friday.core.modes import (
    alerting_node,
    automation_node,
    clarify_node,
    conversation_node,
    device_node,
    research_node,
    routing_node,
    security_lockdown_node,
)
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode

logger = logging.getLogger("friday.core.graph")

# Map a routed Mode to the name of the node that handles it. ROUTING normalizes
# every mode without a dedicated node to CONVERSATION, so this map is total over
# the modes ``node_routing`` can leave on the state.
_MODE_TO_NODE: dict[Mode, str] = {
    Mode.RESEARCH: "research",
    Mode.CLARIFY: "clarify",
    Mode.CONVERSATION: "conversation",
    Mode.AUTOMATION: "automation",
    Mode.DEVICE_CONTROL: "device",
    Mode.ALERTING: "alerting",
    Mode.SECURITY_LOCKDOWN: "security_lockdown",
}


class ModeGraph:
    """A compiled LangGraph mode loop exposing ``async invoke``."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        from langgraph.graph import END, START, StateGraph

        self._routing = routing_node(orchestrator)
        self._conversation = conversation_node(orchestrator)
        self._research = research_node(orchestrator)
        self._clarify = clarify_node(orchestrator)
        self._automation = automation_node(orchestrator)
        self._device = device_node(orchestrator)
        self._alerting = alerting_node(orchestrator)
        self._security_lockdown = security_lockdown_node(orchestrator)

        builder: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(
            GraphState
        )
        builder.add_node("routing", self._routing)
        builder.add_node("conversation", self._conversation)
        builder.add_node("research", self._research)
        builder.add_node("clarify", self._clarify)
        builder.add_node("automation", self._automation)
        builder.add_node("device", self._device)
        builder.add_node("alerting", self._alerting)
        builder.add_node("security_lockdown", self._security_lockdown)

        # Every mode node terminates the turn; the conditional edge fans out from
        # routing to exactly one of them.
        node_names = [
            "conversation",
            "research",
            "clarify",
            "automation",
            "device",
            "alerting",
            "security_lockdown",
        ]

        builder.add_edge(START, "routing")
        builder.add_conditional_edges(
            "routing",
            _select_node,
            {name: name for name in node_names},
        )
        for name in node_names:
            builder.add_edge(name, END)

        self._compiled = builder.compile()

    async def invoke(self, state: GraphState) -> GraphState:
        """Run the graph for one turn and return the advanced state."""
        raw = await self._compiled.ainvoke(state)
        # LangGraph returns the merged channel values as a dict; re-validate into
        # the typed model so callers always receive a GraphState.
        if isinstance(raw, GraphState):
            return raw
        return GraphState.model_validate(raw)


class _StateMachine:
    """Minimal async fallback with the same ``invoke`` contract as :class:`ModeGraph`.

    Used only if LangGraph cannot be imported/compiled. Executes the identical
    flow: routing, then dispatch to one mode node, then end.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._routing = routing_node(orchestrator)
        self._nodes = {
            "conversation": conversation_node(orchestrator),
            "research": research_node(orchestrator),
            "clarify": clarify_node(orchestrator),
            "automation": automation_node(orchestrator),
            "device": device_node(orchestrator),
            "alerting": alerting_node(orchestrator),
            "security_lockdown": security_lockdown_node(orchestrator),
        }

    async def invoke(self, state: GraphState) -> GraphState:
        state = await self._routing(state)
        node_name = _MODE_TO_NODE.get(state.mode, "conversation")
        return await self._nodes[node_name](state)


def _select_node(state: GraphState) -> str:
    """Conditional-edge selector: map the routed mode to its node name."""
    return _MODE_TO_NODE.get(state.mode, "conversation")


def build_graph(orchestrator: Orchestrator) -> ModeGraph | _StateMachine:
    """Build the mode loop, preferring LangGraph and falling back if it breaks.

    Returns an object exposing ``async def invoke(state) -> GraphState``. The
    LangGraph path is primary; any import/compile failure is logged and the
    hand-rolled :class:`_StateMachine` is returned instead so the phase never
    blocks on a framework incompatibility.
    """
    try:
        return ModeGraph(orchestrator)
    except Exception as exc:  # pragma: no cover - resilience seam, not hit on pinned deps
        logger.warning(
            "LangGraph unavailable (%s); falling back to the async state machine.",
            exc,
        )
        return _StateMachine(orchestrator)
