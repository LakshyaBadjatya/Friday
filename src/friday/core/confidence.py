# © Lakshya Badjatya — Author
"""Calibrated confidence scoring: how sure FRIDAY is about a turn's answer.

After the orchestrator has routed, dispatched, and synthesized a turn, several
independent signals say something about how trustworthy the reply is — the
router's own confidence in the classification, a specialist agent's confidence
in its work, whether the answer was *grounded* in retrieved memory, and whether
a research web search actually returned a hit. This module blends those signals
into one calibrated :class:`ConfidenceScore` in ``[0, 1]`` plus a one-line human
rationale, so the orchestrator can append an honest caveat when confidence is
low without re-deriving the arithmetic itself.

The blend is **pure, deterministic, and monotonic** in every positive signal:
raising any one input (router confidence up, agent confidence up, grounding
turned on, a web hit turned on) can only ever raise — never lower — the score.
It is computed as a fixed convex combination of the signals present:

* ``route_confidence`` carries weight ``0.45`` (the router is the backbone
  signal and is always present);
* ``agent_confidence`` carries weight ``0.30`` *when a specialist agent ran*
  (``None`` when no agent dispatched, e.g. plain conversation);
* ``retrieval_grounded`` contributes weight ``0.15`` as a 0/1 indicator (the
  answer cited retrieved memory);
* ``web_search_hit`` contributes weight ``0.10`` as a 0/1 indicator (a research
  search returned at least one source); ``None`` when no search was attempted.

Only the weights of the signals that are actually present are summed, and the
weighted value is divided by that present-weight total — so an absent signal
(``None``) neither helps nor hurts, and the result is a true weighted average
that always lands in ``[0, 1]``. It is clamped to ``[0, 1]`` regardless, so a
mis-calibrated out-of-range input can never escape the bound. ``mode`` is
carried for the rationale only; it does not move the number.

This module lives in ``core/`` and so, like the orchestrator and critic, imports
NO LLM SDK and never reads :func:`friday.config.get_settings` — its inputs arrive
by construction (the orchestrator builds the signals from state and the flag /
threshold are read by ``app.py``). It is offline and side-effect free.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from friday.core.state import GraphState

# Fixed blend weights for each signal (documented in the module docstring). The
# router signal is always present; the others are conditional and only enter the
# weighted average when their value is not ``None``.
_W_ROUTE: float = 0.45
_W_AGENT: float = 0.30
_W_GROUNDED: float = 0.15
_W_WEB: float = 0.10


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0.0, 1.0]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class ConfidenceSignals(BaseModel):
    """The inputs to the confidence blend for one turn (a frozen value object).

    Attributes:
        route_confidence: The router's confidence in its classification
            (``RouteDecision.confidence``), always present, in ``[0, 1]``.
        agent_confidence: A dispatched specialist agent's confidence in its own
            result, or ``None`` when no agent ran (e.g. plain conversation).
        retrieval_grounded: Whether the answer was grounded in retrieved memory
            (the Knowledge agent cited stored ``source_id``s).
        web_search_hit: Whether a research web search returned at least one
            source, or ``None`` when no search was attempted this turn.
        mode: The turn's final :class:`~friday.core.state.Mode` as a string,
            carried for the rationale line only (it does not move the score).
    """

    model_config = ConfigDict(frozen=True)

    route_confidence: float
    agent_confidence: float | None = None
    retrieval_grounded: bool = False
    web_search_hit: bool | None = None
    mode: str


class ConfidenceScore(BaseModel):
    """A calibrated confidence verdict for one turn.

    Attributes:
        value: The blended confidence, always clamped to ``[0, 1]``.
        rationale: A single short, human-readable line explaining the verdict
            (which signals were present and how they landed). Never empty.
    """

    value: float = Field(ge=0.0, le=1.0)
    rationale: str


class ConfidenceScorer:
    """Blends :class:`ConfidenceSignals` into a calibrated :class:`ConfidenceScore`.

    Pure and deterministic: the same signals always produce the same score, with
    no clock, no network, and no settings read. The blend is a fixed convex
    combination over the signals that are present (see the module docstring),
    normalized by the present-weight total so an absent (``None``) signal neither
    helps nor hurts, and the result is monotonic in every positive signal —
    raising one input never lowers the score.
    """

    def score(self, signals: ConfidenceSignals) -> ConfidenceScore:
        """Return the calibrated :class:`ConfidenceScore` for ``signals``.

        Each present signal contributes ``weight * value`` to the numerator and
        ``weight`` to the denominator; the value is ``numerator / denominator``
        (the router signal is always present, so the denominator is never zero).
        The result is clamped to ``[0, 1]`` so an out-of-range input can never
        escape the bound, and a short rationale records which signals fed in.
        """
        # The router signal is always present and anchors the denominator.
        route = _clamp01(signals.route_confidence)
        numerator = _W_ROUTE * route
        denominator = _W_ROUTE
        parts: list[str] = [f"route {route:.2f}"]

        if signals.agent_confidence is not None:
            agent = _clamp01(signals.agent_confidence)
            numerator += _W_AGENT * agent
            denominator += _W_AGENT
            parts.append(f"agent {agent:.2f}")

        # Boolean signals enter as 0/1 indicators: a positive flag adds its full
        # weight to the numerator (monotonic up), while its weight always enters
        # the denominator so "grounded but unsure" reads as a partial average
        # rather than a free pass.
        if signals.retrieval_grounded:
            numerator += _W_GROUNDED
        denominator += _W_GROUNDED
        parts.append("grounded" if signals.retrieval_grounded else "ungrounded")

        if signals.web_search_hit is not None:
            if signals.web_search_hit:
                numerator += _W_WEB
            denominator += _W_WEB
            parts.append("web hit" if signals.web_search_hit else "web miss")

        value = _clamp01(numerator / denominator)
        rationale = (
            f"{signals.mode}: confidence {value:.2f} from " + ", ".join(parts) + "."
        )
        return ConfidenceScore(value=value, rationale=rationale)


def signals_from_state(state: GraphState) -> ConfidenceSignals:
    """Extract :class:`ConfidenceSignals` from a turn's :class:`GraphState`.

    Reads the router decision and the scratchpad keys the orchestrator already
    sets, so callers need not know the internals:

    * ``route_confidence`` <- ``state.route.confidence`` (``0.0`` if unrouted);
    * ``agent_confidence`` <- ``scratchpad["agent_confidence"]`` (present only
      when a specialist / knowledge agent ran);
    * ``retrieval_grounded`` <- truthy ``scratchpad["retrieval_grounded"]`` if a
      grounding flag was stamped, else inferred from a non-empty
      ``scratchpad["citations"]`` list;
    * ``web_search_hit`` <- whether ``scratchpad["web_search_results"]`` holds at
      least one row when a search was invoked, else ``None`` (no search this
      turn);
    * ``mode`` <- ``state.mode``.

    Only well-typed values are accepted off the (``dict[str, Any]``) scratchpad;
    anything of an unexpected type is treated as absent, so a malformed entry can
    never crash the scorer or fabricate confidence.
    """
    scratch: dict[str, Any] = state.scratchpad
    route_confidence = state.route.confidence if state.route is not None else 0.0

    raw_agent = scratch.get("agent_confidence")
    agent_confidence = (
        float(raw_agent) if isinstance(raw_agent, (int, float)) else None
    )

    grounded = bool(scratch.get("retrieval_grounded"))
    if not grounded:
        citations = scratch.get("citations")
        grounded = isinstance(citations, list) and len(citations) > 0

    web_search_hit: bool | None = None
    if scratch.get("web_search_invoked"):
        results = scratch.get("web_search_results")
        web_search_hit = isinstance(results, list) and len(results) > 0

    return ConfidenceSignals(
        route_confidence=route_confidence,
        agent_confidence=agent_confidence,
        retrieval_grounded=grounded,
        web_search_hit=web_search_hit,
        mode=str(state.mode),
    )
