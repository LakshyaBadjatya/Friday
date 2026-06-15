"""Integration tests for the Stage-2 roster wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` (and the FastAPI app via
:func:`friday.app.create_app`) and asserts:

* ``GET /roster`` lists FRIDAY plus the eight specialist personas
  (name/title/scope/namespace).
* A :class:`~friday.roster.RosterRegistry` is built and surfaced on the runtime /
  ``app.state``.
* Addressing a turn by a persona code-name ("GECKO, ...") selects that persona and
  routes under its least-privilege scope + memory namespace; an un-addressed turn
  is unchanged (no persona selected).
* The Stage-2 perception-extra flags (desktop / voiceprint / proactive) default
  OFF, so none of those seams are constructed.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday import app as app_mod
from friday.app import build_runtime, create_app
from friday.config import Settings
from friday.core.state import GraphState
from friday.roster import RosterRegistry

# The canonical roster code-names the listing must contain.
_EXPECTED_NAMES = {
    "FRIDAY",
    "EDITH",
    "ORACLE",
    "GECKO",
    "KAREN",
    "VERONICA",
    "JOCASTA",
    "VISION",
    "FORGE",
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _app(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FastAPI:
    monkeypatch.setattr(app_mod, "get_settings", lambda: settings)
    return create_app()


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #
def test_stage2_flag_defaults() -> None:
    settings = Settings(_env_file=None)
    # The read-only idea-batch tools are ON by default.
    assert settings.enable_extra_tools is True
    # The perception-extra seams default OFF.
    assert settings.enable_desktop is False
    assert settings.enable_voiceprint is False
    assert settings.enable_proactive is False
    # The side-effecting idea-batch tools default OFF.
    assert settings.enable_downloads_butler is False
    assert settings.enable_media_control is False


# --------------------------------------------------------------------------- #
# RosterRegistry is built + surfaced
# --------------------------------------------------------------------------- #
def test_roster_registry_built_on_runtime() -> None:
    runtime = build_runtime(_settings())
    assert isinstance(runtime.roster, RosterRegistry)
    assert set(runtime.roster.names()) == _EXPECTED_NAMES


# --------------------------------------------------------------------------- #
# GET /roster lists the 8 personas + FRIDAY
# --------------------------------------------------------------------------- #
def test_get_roster_lists_all_personas(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch, _settings())
    with TestClient(app) as client:
        resp = client.get("/roster")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 9
        names = {p["name"] for p in body["personas"]}
        assert names == _EXPECTED_NAMES
        # FRIDAY (the prime) leads the declaration-ordered listing.
        assert body["personas"][0]["name"] == "FRIDAY"

        by_name = {p["name"]: p for p in body["personas"]}
        gecko = by_name["GECKO"]
        assert gecko["title"] == "Finance & Markets"
        assert gecko["namespace"] == "gecko"
        # GECKO's least-privilege scope is finance + web research.
        assert "market_data" in gecko["scope"]
        assert "web_search" in gecko["scope"]
        # Scope is sorted for a stable listing.
        assert gecko["scope"] == sorted(gecko["scope"])


# --------------------------------------------------------------------------- #
# Address-by-name routing
# --------------------------------------------------------------------------- #
async def test_addressing_gecko_routes_under_finance_scope() -> None:
    runtime = build_runtime(_settings())
    orch = runtime.orchestrator

    state = GraphState(session_id="s1", user_input="GECKO, what's the price of gold?")
    result = await orch.handle(state)

    # The named persona was selected and its least-privilege finance scope applied.
    selected = result.scratchpad.get("persona")
    assert selected == "GECKO"
    scope = result.scratchpad.get("persona_scope")
    assert scope is not None
    assert "market_data" in scope
    assert "web_search" in scope
    # The turn runs under the persona's own memory namespace.
    assert result.scratchpad.get("persona_namespace") == "gecko"


async def test_addressing_ask_vision_form_selects_persona() -> None:
    runtime = build_runtime(_settings())
    orch = runtime.orchestrator
    state = GraphState(
        session_id="s2", user_input="ask VISION to analyze the latency numbers"
    )
    result = await orch.handle(state)
    assert result.scratchpad.get("persona") == "VISION"
    assert result.scratchpad.get("persona_namespace") == "vision"


async def test_unaddressed_turn_selects_no_persona() -> None:
    runtime = build_runtime(_settings())
    orch = runtime.orchestrator
    state = GraphState(session_id="s3", user_input="what's the weather like?")
    result = await orch.handle(state)
    # No leading persona address -> the hook is inert (unchanged behaviour).
    assert "persona" not in result.scratchpad


def test_lowercase_name_is_not_addressed() -> None:
    # A bare lowercase word that merely happens to match a name must NOT be
    # treated as an address — addressing is by the explicit code-name form.
    import asyncio

    runtime = build_runtime(_settings())
    orch = runtime.orchestrator
    state = GraphState(session_id="s4", user_input="forge a new plan for the week")
    result = asyncio.run(orch.handle(state))
    assert "persona" not in result.scratchpad


# --------------------------------------------------------------------------- #
# Perception-extra flags default off → no seam constructed
# --------------------------------------------------------------------------- #
def test_desktop_voiceprint_proactive_default_off() -> None:
    runtime = build_runtime(_settings())
    assert runtime.desktop is None
    assert runtime.owner_identity is None
    assert runtime.anomaly_detector is None
    assert runtime.foresight is None


def test_desktop_built_when_enabled() -> None:
    from friday.desktop import AuditedDesktop

    runtime = build_runtime(_settings(enable_desktop=True))
    assert isinstance(runtime.desktop, AuditedDesktop)


def test_voiceprint_built_when_enabled() -> None:
    from friday.voice.voiceprint import OwnerIdentity

    runtime = build_runtime(_settings(enable_voiceprint=True))
    assert isinstance(runtime.owner_identity, OwnerIdentity)


def test_proactive_built_when_enabled() -> None:
    from friday.proactive import AnomalyDetector, Foresight

    runtime = build_runtime(_settings(enable_proactive=True))
    assert isinstance(runtime.anomaly_detector, AnomalyDetector)
    assert isinstance(runtime.foresight, Foresight)


# --------------------------------------------------------------------------- #
# AnomalyDetector wired into the scheduler's system_check action
# --------------------------------------------------------------------------- #
async def test_system_check_flags_cpu_spike_when_proactive_on() -> None:
    """A CPU spike across ticks is surfaced via the notify sink as an anomaly.

    Drives the REAL ``system_check`` action the wiring registers: a programmable
    sampler holds CPU flat (well under the 90% breach threshold, so no threshold
    alert fires) and then spikes once. With proactive on, the AnomalyDetector wired
    into the action flags the spike and emits one "System anomaly" message — even
    though the static breach threshold was never crossed.
    """
    from friday.scheduler.store import Trigger
    from friday.system.monitor import SystemMonitor, SystemStats
    from friday.tools.notify import NotifyTool

    class _ProgrammableSampler:
        """One CPU value per *tick*; the action samples twice per tick (check +
        stats), so each tick's value is returned for both of that tick's calls."""

        def __init__(self, cpus_per_tick: list[float]) -> None:
            self._cpus = cpus_per_tick
            self._calls = 0

        def sample(self) -> SystemStats:
            tick = self._calls // 2
            cpu = self._cpus[min(tick, len(self._cpus) - 1)]
            self._calls += 1
            return SystemStats(
                cpu_percent=cpu,
                mem_percent=20.0,
                disk_percent=30.0,
                temp_c=40.0,
                load1=0.5,
                ts="2026-06-15T12:00:00+00:00",
            )

    runtime = build_runtime(_settings(enable_proactive=True))
    detector = runtime.anomaly_detector
    assert detector is not None

    # Low, mildly-varying CPU for the warm-up (so the rolling std is non-zero),
    # then a sharp spike to 80% — under the 90% breach threshold, so ONLY the
    # anomaly path can flag it. One value per tick.
    sampler = _ProgrammableSampler([10.0, 11.0, 9.0, 10.0, 11.0, 80.0])
    monitor = SystemMonitor(sampler, cpu_threshold=90.0)
    notify = NotifyTool()

    action = app_mod._make_system_check_action(monitor, notify, detector)  # noqa: SLF001
    trigger = Trigger(
        id=1, name="t", kind="interval", spec="60", action="system_check"
    )
    for _ in range(6):
        await action(trigger)

    subjects = [m.subject for m in notify.sink]
    bodies = [m.body for m in notify.sink]
    # No threshold breach ever fired (CPU stayed <= 80 < 90).
    assert "System alert" not in subjects
    # The spike was flagged as an anomaly.
    assert "System anomaly" in subjects
    assert any("spike" in b.lower() for b in bodies)


async def test_system_check_no_anomaly_when_proactive_off() -> None:
    """With proactive off the action never flags a spike (behaviour unchanged)."""
    from friday.scheduler.store import Trigger
    from friday.system.monitor import SystemMonitor, SystemStats
    from friday.tools.notify import NotifyTool

    class _SpikySampler:
        def __init__(self) -> None:
            self._cpus = [10.0, 10.0, 10.0, 10.0, 10.0, 80.0]
            self._i = 0

        def sample(self) -> SystemStats:
            cpu = self._cpus[min(self._i, len(self._cpus) - 1)]
            self._i += 1
            return SystemStats(
                cpu_percent=cpu,
                mem_percent=20.0,
                disk_percent=30.0,
                temp_c=40.0,
                load1=0.5,
                ts="2026-06-15T12:00:00+00:00",
            )

    monitor = SystemMonitor(_SpikySampler(), cpu_threshold=90.0)
    notify = NotifyTool()
    action = app_mod._make_system_check_action(monitor, notify, None)  # noqa: SLF001
    trigger = Trigger(
        id=1, name="t", kind="interval", spec="60", action="system_check"
    )
    for _ in range(6):
        await action(trigger)
    assert notify.sink == []
