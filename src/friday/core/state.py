"""Core graph state: :class:`Mode`, :class:`RouteDecision`, :class:`GraphState`.

This module is the single source of truth for FRIDAY's mode loop state. It is
deliberately dependency-light — it imports only :class:`friday.providers.llm.Message`
for the conversation buffer and otherwise depends on nothing in ``core`` so it
can be imported freely by ``router.py``, ``orchestrator.py``, ``modes.py``, and
``graph.py`` without creating an import cycle.

:class:`RouteDecision` lives here (not in ``router.py``) precisely so that
``router.py`` can import it from ``state`` while ``state`` never needs to import
``router`` — keeping the dependency edge one-directional.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from friday.providers.llm import Message


class Mode(StrEnum):
    """The operating mode of the core loop.

    Voice/security modes are deferred to later phases; this phase defines the
    minimal set needed for the text core loop.
    """

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    ROUTING = "ROUTING"
    CONVERSATION = "CONVERSATION"
    RESEARCH = "RESEARCH"
    CLARIFY = "CLARIFY"


class RouteDecision(BaseModel):
    """The router's classification of a single user turn.

    Defined in ``state`` (not ``router``) to avoid a router <-> state import
    cycle: ``router`` imports this from here.
    """

    mode: Mode
    agent: str | None = None
    rationale: str
    confidence: float


class GraphState(BaseModel):
    """Mutable state threaded through the mode-loop graph for one turn.

    Round-trips losslessly through ``model_dump_json`` / ``model_validate_json``
    so it can be persisted or transported between graph steps.
    """

    session_id: str
    mode: Mode = Mode.IDLE
    messages: list[Message] = Field(default_factory=list)
    user_input: str
    route: RouteDecision | None = None
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    response: str | None = None
