"""Per-persona free-model assignment + the orchestrator's use of it.

Two layers are covered with zero network:

* :func:`friday.models.personas.resolve_persona_models` — the greedy, distinct,
  availability-scoped assignment (full catalog, single-provider fall-back, empty);
* :class:`friday.core.orchestrator.Orchestrator` — that addressing a specialist
  (a) answers in that specialist's voice (its charter is injected, not the prime's
  persona file) and (b) routes the turn through that specialist's model, while an
  un-addressed turn stays on the prime/default; plus the opt-in auto-delegate.

Async tests need no marker: the project runs pytest-asyncio in ``asyncio_mode =
"auto"`` (see pyproject.toml).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.config import get_settings
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.memory.short_term import ShortTermMemory
from friday.models.catalog import DEFAULT_CATALOG, ModelCatalog, ModelInfo
from friday.models.gateway import ModelGateway
from friday.models.personas import (
    PERSONA_MODEL_PREFERENCES,
    resolve_persona_models,
)
from friday.providers.llm import (
    FakeLLM,
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)
from friday.roster import ROSTER, RosterRegistry
from friday.roster.custom import merge_personas
from friday.roster.definitions import ROSTER_PERSONAS, Persona
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)

_ALL_PROVIDERS = {"openrouter", "opencode", "nvidia"}


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


# --------------------------------------------------------------------------- #
# resolve_persona_models — assignment over the real catalog
# --------------------------------------------------------------------------- #
def test_every_persona_gets_a_distinct_model_when_all_providers_wired() -> None:
    """With all three free providers available, all nine personas are distinct."""
    catalog = ModelCatalog(available_providers=_ALL_PROVIDERS)
    assigned = resolve_persona_models(catalog)
    # One model per persona...
    assert set(assigned) == {p.name for p in ROSTER_PERSONAS}
    # ...and no model assigned twice.
    assert len(set(assigned.values())) == len(assigned)
    # Every assigned id is a real, available catalog id.
    available = set(catalog.ids())
    assert all(model_id in available for model_id in assigned.values())


def test_assignment_respects_role_matched_primary_when_free() -> None:
    """Each persona gets its first-choice model when nothing else has taken it."""
    catalog = ModelCatalog(available_providers=_ALL_PROVIDERS)
    assigned = resolve_persona_models(catalog)
    # FORGE's coder model is unique to it, so it always lands its primary.
    assert assigned["FORGE"] == PERSONA_MODEL_PREFERENCES["FORGE"][0]
    assert assigned["FRIDAY"] == PERSONA_MODEL_PREFERENCES["FRIDAY"][0]


def test_single_provider_assigns_distinct_subset_and_omits_the_rest() -> None:
    """OpenRouter-only: ids stay distinct; personas with no free id are omitted."""
    catalog = ModelCatalog(available_providers={"openrouter"})
    assigned = resolve_persona_models(catalog)
    available = set(catalog.ids())
    # Only seven OpenRouter free models exist, so at most seven personas assigned.
    assert len(assigned) <= len(available)
    assert len(set(assigned.values())) == len(assigned)  # still distinct
    assert all(model_id in available for model_id in assigned.values())
    # Nine personas, seven models -> at least two omitted (fall back to default).
    assert len(assigned) < len(ROSTER_PERSONAS)


def test_empty_catalog_assigns_nothing() -> None:
    """No available providers -> no assignments (everyone uses the default model)."""
    catalog = ModelCatalog(available_providers=set())
    assert resolve_persona_models(catalog) == {}


def test_default_catalog_has_enough_free_models_for_distinct_assignment() -> None:
    """Guard: the shipped catalog can serve nine distinct models (>= persona count)."""
    free_ids = {m.id for m in DEFAULT_CATALOG if m.free}
    assert len(free_ids) >= len(ROSTER_PERSONAS)


# --------------------------------------------------------------------------- #
# Orchestrator — a recording LLM to inspect the injected system prompt
# --------------------------------------------------------------------------- #
class _RecordingLLM(LLMProvider):
    """Captures each call's messages + model kwarg, returns a fixed reply."""

    def __init__(self, reply: str = "ack") -> None:
        self.calls: list[tuple[list[Message], str | None]] = []
        self._reply = reply

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), model))
        return _resp(self._reply)

    @property
    def last_system_prompt(self) -> str:
        messages, _ = self.calls[-1]
        return next(m.content or "" for m in messages if m.role == "system")


def _orch(
    llm: LLMProvider, *, roster: RosterRegistry = ROSTER, **kwargs: object
) -> Orchestrator:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    return Orchestrator(
        llm=llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        roster=roster,
        **kwargs,  # type: ignore[arg-type]
    )


def _custom_persona(name: str = "NOVA") -> Persona:
    """A user-declared custom operator (the shape merge_personas folds in)."""
    return Persona(
        name=name,
        title="Operations",
        allowed_tools=frozenset({"notify"}),
        memory_namespace=name.lower(),
        system_prompt=f"You are {name}, the custom operations operator.",
    )


async def test_addressing_specialist_injects_its_charter_not_the_prime() -> None:
    """Addressing EDITH puts EDITH's charter in the system prompt, not friday.md."""
    llm = _RecordingLLM("Secured, Boss.")
    orch = _orch(llm)
    state = GraphState(session_id="s1", user_input="EDITH, what's 2+2")
    await orch.handle(state)
    prompt = llm.last_system_prompt
    assert "EDITH" in prompt
    assert "security operator" in prompt  # from EDITH's roster system_prompt
    assert state.scratchpad.get("persona") == "EDITH"


async def test_unaddressed_turn_uses_prime_persona_file() -> None:
    """An un-addressed turn keeps the prime path (friday.md), no specialist charter."""
    llm = _RecordingLLM("Four, Boss.")
    orch = _orch(llm)
    state = GraphState(session_id="s2", user_input="what's 2+2")
    await orch.handle(state)
    prompt = llm.last_system_prompt
    assert "FRIDAY — Persona Specification" in prompt  # the persona file header
    assert state.scratchpad.get("persona") is None


# --------------------------------------------------------------------------- #
# Orchestrator — a real gateway to verify per-persona model routing
# --------------------------------------------------------------------------- #
_TWO_MODEL_CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="openrouter:google/gemma-4-31b-it:free",
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
        label="Gemma 4 31B IT",
        free=True,
    ),
    ModelInfo(
        id="opencode:mimo-v2.5-free",
        provider="opencode",
        model="mimo-v2.5-free",
        label="MiMo v2.5",
        free=True,
    ),
)
_DEFAULT_MODEL = "openrouter:google/gemma-4-31b-it:free"
_EDITH_MODEL = "opencode:mimo-v2.5-free"


def _gateway(
    *, openrouter: list[LLMResponse], opencode: list[LLMResponse]
) -> ModelGateway:
    catalog = ModelCatalog(
        available_providers={"openrouter", "opencode"}, catalog=_TWO_MODEL_CATALOG
    )
    providers: dict[str, LLMProvider] = {
        "openrouter": FakeLLM(responses=openrouter),
        "opencode": FakeLLM(responses=opencode),
    }
    return ModelGateway(providers, catalog, default_model_id=_DEFAULT_MODEL)


async def test_addressed_specialist_runs_on_its_own_model() -> None:
    """Addressing EDITH routes the turn through EDITH's assigned (opencode) model."""
    gateway = _gateway(
        openrouter=[_resp("prime on gemma")],
        opencode=[_resp("EDITH on MiMo")],
    )
    orch = _orch(gateway, persona_models={"EDITH": _EDITH_MODEL})
    state = GraphState(session_id="s3", user_input="EDITH, what's 2+2")
    await orch.handle(state)
    assert state.response == "EDITH on MiMo"
    # Per-call routing never mutates the gateway's active model.
    assert gateway.active_model_id == _DEFAULT_MODEL


async def test_unaddressed_turn_runs_on_default_model() -> None:
    """With no persona, the turn stays on the gateway's active/default model."""
    gateway = _gateway(
        openrouter=[_resp("prime on gemma")],
        opencode=[_resp("should not be used")],
    )
    orch = _orch(gateway, persona_models={"EDITH": _EDITH_MODEL})
    state = GraphState(session_id="s4", user_input="what's 2+2")
    await orch.handle(state)
    assert state.response == "prime on gemma"


async def test_explicit_model_override_beats_persona_model() -> None:
    """An explicit per-turn override wins over the addressed persona's model."""
    gateway = _gateway(
        openrouter=[_resp("override on gemma")],
        opencode=[_resp("EDITH on MiMo")],
    )
    orch = _orch(gateway, persona_models={"EDITH": _EDITH_MODEL})
    # Address EDITH (persona model = opencode) but override to openrouter for the turn.
    state = GraphState(
        session_id="s5",
        user_input="EDITH, what's 2+2",
        model_override=_DEFAULT_MODEL,
    )
    await orch.handle(state)
    assert state.response == "override on gemma"


# --------------------------------------------------------------------------- #
# Orchestrator — opt-in auto-delegate by topic
# --------------------------------------------------------------------------- #
@pytest.fixture
def _auto_delegate_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRIDAY_ENABLE_AUTO_DELEGATE", "true")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("FRIDAY_ENABLE_AUTO_DELEGATE", raising=False)
    get_settings.cache_clear()


async def test_auto_delegate_off_keeps_prime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the flag off, a topical turn that names no operator stays with the prime."""
    # Force the flag off explicitly (hermetic: independent of any local .env value).
    monkeypatch.setenv("FRIDAY_ENABLE_AUTO_DELEGATE", "false")
    get_settings.cache_clear()
    try:
        llm = _RecordingLLM("ok")
        orch = _orch(llm)
        state = GraphState(session_id="s6", user_input="help me schedule a reminder")
        await orch.handle(state)
        assert state.scratchpad.get("persona") is None
    finally:
        get_settings.cache_clear()


async def test_auto_delegate_routes_by_topic_when_enabled(_auto_delegate_on) -> None:
    """With the flag on, a scheduling turn auto-delegates to ORACLE (automation)."""
    llm = _RecordingLLM("scheduled")
    orch = _orch(llm)
    state = GraphState(session_id="s7", user_input="set up a scheduler job for me")
    await orch.handle(state)
    assert state.scratchpad.get("persona") == "ORACLE"


async def test_explicit_address_beats_auto_delegate(_auto_delegate_on) -> None:
    """An explicit address wins even when an auto-delegate keyword is also present."""
    llm = _RecordingLLM("ok")
    orch = _orch(llm)
    # "schedule" would auto-delegate to ORACLE, but VISION is addressed explicitly.
    state = GraphState(
        session_id="s8", user_input="VISION, look into the schedule for me"
    )
    await orch.handle(state)
    assert state.scratchpad.get("persona") == "VISION"


# --------------------------------------------------------------------------- #
# Custom operators work the same as the built-ins
# --------------------------------------------------------------------------- #
def test_custom_operator_gets_a_distinct_leftover_model() -> None:
    """A merged custom operator is assigned its own model, distinct from all others."""
    catalog = ModelCatalog(available_providers=_ALL_PROVIDERS)
    personas = (*ROSTER_PERSONAS, _custom_persona("NOVA"))
    assigned = resolve_persona_models(catalog, personas)
    assert "NOVA" in assigned  # custom operators are covered, not just built-ins
    # NOVA's model differs from every built-in's, and all assignments stay distinct.
    others = {m for n, m in assigned.items() if n != "NOVA"}
    assert assigned["NOVA"] not in others
    assert len(set(assigned.values())) == len(assigned)


async def test_custom_operator_runs_on_its_own_model() -> None:
    """Addressing a custom operator routes the turn through its assigned model."""
    roster = RosterRegistry(merge_personas(ROSTER_PERSONAS, [_custom_persona("NOVA")]))
    gateway = _gateway(
        openrouter=[_resp("prime on gemma")],
        opencode=[_resp("NOVA on MiMo")],
    )
    orch = _orch(gateway, roster=roster, persona_models={"NOVA": _EDITH_MODEL})
    state = GraphState(session_id="c1", user_input="NOVA, what's 2+2")
    await orch.handle(state)
    assert state.response == "NOVA on MiMo"
    assert state.scratchpad.get("persona") == "NOVA"


async def test_custom_operator_answers_in_its_own_voice() -> None:
    """A custom operator's charter is injected as the system prompt, like a built-in."""
    roster = RosterRegistry(merge_personas(ROSTER_PERSONAS, [_custom_persona("NOVA")]))
    llm = _RecordingLLM("On it, Boss.")
    orch = _orch(llm, roster=roster)
    state = GraphState(session_id="c2", user_input="NOVA, what's 2+2")
    await orch.handle(state)
    prompt = llm.last_system_prompt
    assert "NOVA, the custom operations operator" in prompt
    assert state.scratchpad.get("persona") == "NOVA"
