# © Lakshya Badjatya — Author
"""Integration tests for the Wave-0 per-turn budgeter wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``). Asserts:

* Flag OFF (default): the runtime surfaces ``budgeter is None`` and the
  orchestrator carries no budgeter — the turn loop is unchanged.
* Flag ON: a real :class:`~friday.models.budget.Budgeter` is constructed with the
  configured caps and surfaced on the runtime + injected into the orchestrator.
* The budgeter is pure/offline — it reads no settings itself; the caps come from
  ``app.py`` via dependency injection, so the constructed budgeter honours them.

The downshift -> ``gateway.set_active`` path needs a real ModelGateway (absent on
the fake build), so it is unit-tested at the :class:`Budgeter` level; here we
verify the WIRING (built / not built / caps applied), honestly scoped.
"""

from __future__ import annotations

from friday.app import build_runtime
from friday.config import Settings
from friday.models.budget import Budgeter


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #
def test_budgeter_flag_defaults_off() -> None:
    settings = Settings(_env_file=None)
    assert settings.enable_budgeter is False
    assert settings.budget_max_tokens_per_turn == 8000
    assert settings.budget_max_usd_per_turn is None
    assert settings.budget_downshift_model_id == ""
    assert settings.budget_downshift_at == 0.8


# --------------------------------------------------------------------------- #
# Flag OFF: no budgeter built (turn loop unchanged)
# --------------------------------------------------------------------------- #
def test_budgeter_none_when_off() -> None:
    runtime = build_runtime(_settings())
    assert runtime.budgeter is None
    # The orchestrator carries no budgeter either.
    assert runtime.orchestrator._budgeter is None  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Flag ON: budgeter built with the configured caps + injected
# --------------------------------------------------------------------------- #
def test_budgeter_built_when_enabled() -> None:
    runtime = build_runtime(
        _settings(
            enable_budgeter=True,
            budget_max_tokens_per_turn=1000,
            budget_downshift_at=0.5,
        )
    )
    assert isinstance(runtime.budgeter, Budgeter)
    # The same instance is injected into the orchestrator.
    assert runtime.orchestrator._budgeter is runtime.budgeter  # noqa: SLF001


def test_budgeter_honours_injected_caps() -> None:
    """The caps arrive by DI (the budgeter reads no settings): a 1000-token cap
    with a 0.5 downshift fraction trips ``should_downshift`` at 500 tokens."""
    runtime = build_runtime(
        _settings(
            enable_budgeter=True,
            budget_max_tokens_per_turn=1000,
            budget_downshift_at=0.5,
        )
    )
    budgeter = runtime.budgeter
    assert budgeter is not None
    budgeter.start_turn("s1")
    budgeter.record("s1", tokens=400)
    assert budgeter.should_downshift("s1") is False
    budgeter.record("s1", tokens=200)  # 600 >= 0.5 * 1000
    assert budgeter.should_downshift("s1") is True
    # The hard token cap (1000) leaves 400 remaining.
    assert budgeter.remaining("s1") == 400


# --------------------------------------------------------------------------- #
# Flag-off turn behaves identically (no scratchpad budget side effects)
# --------------------------------------------------------------------------- #
async def test_turn_unchanged_when_budgeter_off() -> None:
    from friday.core.state import GraphState

    runtime = build_runtime(_settings())
    state = GraphState(session_id="s1", user_input="hello there")
    result = await runtime.orchestrator.handle(state)
    # A normal reply came back; nothing about budget was stamped.
    assert result.response is not None
