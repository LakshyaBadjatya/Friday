# © Lakshya Badjatya — Author
"""Unit tests for task-decomposition: Plan ordering/render + Planner (FakeLLM)."""

from __future__ import annotations

import json

import pytest

from friday.core.planner import Plan, Planner, PlanStep
from friday.providers.llm import FakeLLM, LLMResponse


def _step(sid: str, *, deps: list[str] | None = None, **kw: object) -> PlanStep:
    kw.setdefault("description", sid)
    return PlanStep(id=sid, depends_on=deps or [], **kw)  # type: ignore[arg-type]


def test_topological_order_linear() -> None:
    plan = Plan(goal="g", steps=[_step("a"), _step("b", deps=["a"]), _step("c", deps=["b"])])
    assert [s.id for s in plan.topological_order()] == ["a", "b", "c"]


def test_topological_order_diamond_is_deterministic() -> None:
    # a -> b, a -> c, (b,c) -> d. Ties broken by declared order (b before c).
    plan = Plan(
        goal="g",
        steps=[
            _step("a"),
            _step("b", deps=["a"]),
            _step("c", deps=["a"]),
            _step("d", deps=["b", "c"]),
        ],
    )
    order = [s.id for s in plan.topological_order()]
    assert order[0] == "a" and order[-1] == "d"
    assert order.index("b") < order.index("c")  # declared-order tiebreak


def test_cycle_is_rejected() -> None:
    plan = Plan(goal="g", steps=[_step("a", deps=["b"]), _step("b", deps=["a"])])
    with pytest.raises(ValueError, match="cycle"):
        plan.topological_order()


def test_missing_dependency_is_rejected() -> None:
    plan = Plan(goal="g", steps=[_step("a", deps=["ghost"])])
    with pytest.raises(ValueError, match="unknown step"):
        plan.topological_order()


def test_render_marks_side_effects_and_deps() -> None:
    plan = Plan(
        goal="ship it",
        steps=[
            _step("a", operator="VISION", description="research"),
            _step("b", deps=["a"], operator="FORGE", description="deploy",
                  tool="run_command", side_effecting=True),
        ],
    )
    text = plan.render()
    assert "Plan for: ship it" in text
    assert "[VISION] research" in text
    assert "via run_command" in text
    assert "real-world action" in text
    assert "after a" in text


async def test_decompose_parses_json_steps() -> None:
    payload = json.dumps(
        [
            {"id": "s1", "description": "look it up", "operator": "VISION",
             "tool": "web_search", "depends_on": [], "side_effecting": False},
            {"id": "s2", "description": "notify", "operator": "KAREN",
             "tool": "notify", "depends_on": ["s1"], "side_effecting": True},
        ]
    )
    llm = FakeLLM(responses=[LLMResponse(text=f"Here is the plan:\n{payload}")])
    plan = await Planner(llm).decompose("research then notify")

    assert [s.id for s in plan.steps] == ["s1", "s2"]
    assert plan.steps[1].side_effecting is True
    assert [s.id for s in plan.topological_order()] == ["s1", "s2"]


async def test_decompose_unparseable_falls_back_to_single_step() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="sorry, no JSON here")])
    plan = await Planner(llm).decompose("do the thing")
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "do the thing"


async def test_decompose_provider_error_falls_back() -> None:
    llm = FakeLLM(responses=[])  # exhausted -> ProviderError -> fallback
    plan = await Planner(llm).decompose("do the thing")
    assert len(plan.steps) == 1
    assert plan.steps[0].id == "s1"
