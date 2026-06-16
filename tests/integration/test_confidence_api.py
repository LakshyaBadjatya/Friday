# © Lakshya Badjatya — Author
"""Integration tests for the Wave-0 confidence-scorer wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``). Asserts:

* Flag OFF (default): the runtime surfaces ``confidence is None`` and a real turn
  stamps NO ``scratchpad["confidence"]`` and appends no caveat — behaviour
  unchanged.
* Flag ON: a real :class:`~friday.core.confidence.ConfidenceScorer` is built and
  injected into the orchestrator, and a synthesized turn stamps
  ``scratchpad["confidence"]`` (a value in [0, 1] plus a rationale).
* The scorer reads no settings itself; the flag + threshold are read by the
  orchestrator (DI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import friday.config as config
import friday.core.orchestrator as orch_mod
from friday.app import build_runtime
from friday.config import Settings
from friday.core.confidence import ConfidenceScorer
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


def _set_flag(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool, threshold: float = 0.45
) -> None:
    """Force ``enable_confidence`` + threshold for both config + orchestrator."""
    settings = config.Settings(
        _env_file=None,
        enable_confidence=enabled,
        confidence_note_threshold=threshold,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)


def _orchestrator(synth_llm: FakeLLM, scorer: ConfidenceScorer | None) -> Orchestrator:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    return Orchestrator(
        llm=synth_llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        confidence=scorer,
    )


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #
def test_confidence_flag_defaults_off() -> None:
    settings = Settings(_env_file=None)
    assert settings.enable_confidence is False
    assert settings.confidence_note_threshold == 0.45


# --------------------------------------------------------------------------- #
# Flag OFF: no scorer built, no scratchpad stamp
# --------------------------------------------------------------------------- #
def test_confidence_none_when_off() -> None:
    runtime = build_runtime(_settings())
    assert runtime.confidence is None
    assert runtime.orchestrator._confidence is None  # noqa: SLF001


async def test_turn_stamps_no_confidence_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_flag(monkeypatch, enabled=False)
    synth_llm = FakeLLM(responses=[_resp("All quiet, Boss.")])
    orch = _orchestrator(synth_llm, ConfidenceScorer())
    state = GraphState(session_id="s1", user_input="what's 2+2")
    result = await orch.handle(state)
    assert result.response == "All quiet, Boss."
    # Flag off -> nothing stamped, behaviour unchanged.
    assert "confidence" not in result.scratchpad


async def test_turn_stamps_no_confidence_when_scorer_unwired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag on but no scorer injected: the hook is inert (no stamp).
    _set_flag(monkeypatch, enabled=True)
    synth_llm = FakeLLM(responses=[_resp("All quiet, Boss.")])
    orch = _orchestrator(synth_llm, None)
    state = GraphState(session_id="s1b", user_input="what's 2+2")
    result = await orch.handle(state)
    assert result.response == "All quiet, Boss."
    assert "confidence" not in result.scratchpad


# --------------------------------------------------------------------------- #
# Flag ON: scorer built + injected; a turn stamps a confidence score
# --------------------------------------------------------------------------- #
def test_confidence_built_when_enabled() -> None:
    runtime = build_runtime(_settings(enable_confidence=True))
    assert isinstance(runtime.confidence, ConfidenceScorer)
    assert runtime.orchestrator._confidence is runtime.confidence  # noqa: SLF001


async def test_turn_stamps_confidence_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A low threshold (0.0) keeps the caveat off so we can assert the stamp alone.
    _set_flag(monkeypatch, enabled=True, threshold=0.0)
    synth_llm = FakeLLM(responses=[_resp("Four, Boss. Basic arithmetic holds.")])
    orch = _orchestrator(synth_llm, ConfidenceScorer())
    state = GraphState(session_id="s2", user_input="what's 2+2")
    result = await orch.handle(state)
    assert result.response == "Four, Boss. Basic arithmetic holds."
    # A confidence verdict was stamped after synthesis.
    score = result.scratchpad.get("confidence")
    assert isinstance(score, dict)
    assert 0.0 <= score["value"] <= 1.0
    assert isinstance(score["rationale"], str)
    assert score["rationale"]


async def test_low_confidence_appends_caveat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high threshold forces the (route-only, ungrounded) blend below it, so the
    orchestrator appends its honest one-line caveat to a real reply."""
    _set_flag(monkeypatch, enabled=True, threshold=0.99)
    synth_llm = FakeLLM(responses=[_resp("Four, Boss. Basic arithmetic holds.")])
    orch = _orchestrator(synth_llm, ConfidenceScorer())
    state = GraphState(session_id="s3", user_input="what's 2+2")
    result = await orch.handle(state)
    assert result.response is not None
    assert result.response.startswith("Four, Boss. Basic arithmetic holds.")
    assert "Confidence is on the low side" in result.response
