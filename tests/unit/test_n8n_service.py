"""Unit tests for :class:`friday.n8n.service.N8nService` (offline).

n8n liveness/import is ``respx``-mocked; the docker auto-start is a monkeypatched
:func:`asyncio.create_subprocess_exec` (NO real process spawned); drafting runs on
a scripted :class:`~friday.providers.llm.FakeLLM`. No network, no key leaks.

Covered:
* n8n DOWN + not confirmed -> ``needs_confirmation`` and NO subprocess, NO draft.
* n8n DOWN + confirmed -> runs ``start_cmd`` (mocked) argv-only, then drafts.
* n8n UP + API key (respx-mocked import) -> ``imported`` True.
* n8n UP + API key but import fails -> ``imported`` False with ``import_error``,
  still returns the drafted JSON.
* n8n UP + no API key -> draft only (``imported`` False, no import attempted).
* The API key never appears in the client's repr/str.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

import friday.n8n.service as service_mod
from friday.n8n.client import N8nClient
from friday.n8n.drafter import WorkflowDrafter
from friday.n8n.service import N8nService
from friday.providers.llm import FakeLLM, LLMResponse, Usage

BASE_URL = "http://localhost:5678"
START_CMD = ["docker", "compose", "-f", "docker-compose.yml", "up", "-d", "n8n"]

_VALID_DRAFT = json.dumps(
    {
        "workflow": {
            "name": "My Flow",
            "nodes": [
                {
                    "id": "t",
                    "name": "Start",
                    "type": "n8n-nodes-base.manualTrigger",
                    "position": [250, 300],
                    "parameters": {},
                }
            ],
            "connections": {},
        },
        "setup_notes": ["Wire up the rest"],
    }
)


def _drafter() -> WorkflowDrafter:
    return WorkflowDrafter(
        FakeLLM(responses=[LLMResponse(text=_VALID_DRAFT, tool_calls=[], usage=Usage())])
    )


def _empty_drafter() -> WorkflowDrafter:
    """A drafter whose LLM script is EMPTY — calling it would raise ProviderError.

    Used to prove the no-confirmation path never drafts (the empty script would
    fail loudly if it did).
    """
    return WorkflowDrafter(FakeLLM(responses=[]))


class _FakeProcess:
    """A stand-in for the docker-start subprocess (argv-only, no real spawn)."""

    def __init__(self) -> None:
        self.returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"", b"")


@respx.mock
async def test_down_and_unconfirmed_needs_confirmation_no_start_no_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both health probes fail -> n8n is down.
    respx.get(f"{BASE_URL}/healthz").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{BASE_URL}/rest/login").mock(side_effect=httpx.ConnectError("down"))

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no subprocess may run without confirmation")

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", _boom)

    client = N8nClient(BASE_URL, api_key=None)
    # Empty drafter: if the service drafted, the FakeLLM would raise.
    service = N8nService(client, _empty_drafter(), start_cmd=START_CMD)

    result = await service.make_workflow("post to Slack", confirmed=False)

    assert result["kind"] == "needs_confirmation"
    assert result["action"] == "start_n8n"
    assert "docker" in result["message"]


@respx.mock
async def test_down_and_confirmed_runs_start_cmd_then_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(f"{BASE_URL}/healthz").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{BASE_URL}/rest/login").mock(side_effect=httpx.ConnectError("down"))

    spawned: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        spawned["argv"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", _fake_exec)

    client = N8nClient(BASE_URL, api_key=None)
    service = N8nService(client, _drafter(), start_cmd=START_CMD)

    result = await service.make_workflow("post to Slack", confirmed=True)

    # The argv list was passed positionally (no shell string).
    assert spawned["argv"] == START_CMD
    assert result["kind"] == "workflow"
    assert result["started"] is True
    # It then drafted (no API key -> not imported).
    assert result["imported"] is False
    assert result["workflow"]["name"] == "My Flow"
    assert result["setup_notes"] == ["Wire up the rest"]


@respx.mock
async def test_up_with_api_key_imports_workflow() -> None:
    respx.get(f"{BASE_URL}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    import_route = respx.post(f"{BASE_URL}/api/v1/workflows").mock(
        return_value=httpx.Response(
            200, json={"id": "abc123", "name": "My Flow", "active": False}
        )
    )

    client = N8nClient(BASE_URL, api_key="n8n-secret-key")
    service = N8nService(client, _drafter(), start_cmd=START_CMD)

    result = await service.make_workflow("post to Slack", confirmed=False)

    assert result["kind"] == "workflow"
    assert result["imported"] is True
    assert result["started"] is False
    # The imported body replaces the drafted one (carries the created id).
    assert result["workflow"]["id"] == "abc123"
    # The API key was sent as the X-N8N-API-KEY header (and only there).
    sent = import_route.calls.last.request
    assert sent.headers["X-N8N-API-KEY"] == "n8n-secret-key"


@respx.mock
async def test_up_with_api_key_import_error_is_best_effort() -> None:
    respx.get(f"{BASE_URL}/healthz").mock(return_value=httpx.Response(200))
    respx.post(f"{BASE_URL}/api/v1/workflows").mock(
        return_value=httpx.Response(500, text="boom")
    )

    client = N8nClient(BASE_URL, api_key="n8n-secret-key")
    service = N8nService(client, _drafter(), start_cmd=START_CMD)

    result = await service.make_workflow("post to Slack", confirmed=False)

    # Import failed, but the drafted JSON is still returned.
    assert result["imported"] is False
    assert "import_error" in result
    assert "500" in result["import_error"]
    assert result["workflow"]["name"] == "My Flow"


@respx.mock
async def test_up_without_api_key_drafts_only() -> None:
    respx.get(f"{BASE_URL}/healthz").mock(return_value=httpx.Response(200))
    # No import route registered: if the service tried to import without a key it
    # would raise before the network, so the absence proves no import is attempted.

    client = N8nClient(BASE_URL, api_key=None)
    service = N8nService(client, _drafter(), start_cmd=START_CMD)

    result = await service.make_workflow("post to Slack", confirmed=False)

    assert result["imported"] is False
    assert "import_error" not in result
    assert result["workflow"]["name"] == "My Flow"


def test_api_key_never_in_client_repr() -> None:
    client = N8nClient(BASE_URL, api_key="super-secret-n8n-key")
    assert "super-secret-n8n-key" not in repr(client)
    assert "super-secret-n8n-key" not in str(client)


async def test_start_is_confirm_gated() -> None:
    client = N8nClient(BASE_URL, api_key=None)
    service = N8nService(client, _empty_drafter(), start_cmd=START_CMD)

    # Unconfirmed start is a no-op (no subprocess; would not even reach exec).
    assert await service.start(confirmed=False) is False


# --------------------------------------------------------------------------- #
# Orchestrator hook (the "make a workflow on n8n <X>" intent)
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

import friday.core.orchestrator as orch_mod  # noqa: E402
from friday.core.orchestrator import Orchestrator  # noqa: E402
from friday.core.state import GraphState, Mode  # noqa: E402
from friday.memory.short_term import ShortTermMemory  # noqa: E402
from friday.tools.registry import ToolRegistry  # noqa: E402

_PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


class _StubN8nService:
    """A stand-in n8n service capturing the description + confirm flag it sees."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[tuple[str, bool]] = []

    async def make_workflow(
        self, description: str, *, confirmed: bool = False
    ) -> dict[str, Any]:
        self.calls.append((description, confirmed))
        return self._result


def _orch(n8n: _StubN8nService, *, llm_script: list[str] | None = None) -> Orchestrator:
    registry = ToolRegistry()
    responses = [
        LLMResponse(text=text, tool_calls=[], usage=Usage())
        for text in (llm_script or [])
    ]
    return Orchestrator(
        llm=FakeLLM(responses=responses),
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=_PERSONA_PATH,
        n8n_service=n8n,
    )


async def test_orchestrator_n8n_hook_drafts_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orch_mod, "get_settings", lambda: _N8nEnabledSettings(enable_n8n=True)
    )
    n8n = _StubN8nService(
        {
            "kind": "workflow",
            "imported": True,
            "started": False,
            "workflow": {"name": "Slack flow", "nodes": [], "connections": {}},
            "setup_notes": ["Add Slack credentials"],
        }
    )
    orch = _orch(n8n)
    state = GraphState(
        session_id="n1",
        user_input="Make a workflow on n8n that posts new emails to Slack.",
    )

    out = await orch.handle(state)

    # The description was extracted and passed through with the confirm flag.
    assert n8n.calls == [("posts new emails to Slack", False)]
    assert out.mode is Mode.AUTOMATION
    assert out.response is not None
    assert "Slack flow" in out.response
    assert "Add Slack credentials" in out.response


async def test_orchestrator_n8n_hook_needs_confirmation_surfaces_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orch_mod, "get_settings", lambda: _N8nEnabledSettings(enable_n8n=True)
    )
    n8n = _StubN8nService(
        {
            "kind": "needs_confirmation",
            "action": "start_n8n",
            "message": "n8n isn't running; start it with docker?",
        }
    )
    orch = _orch(n8n)
    state = GraphState(
        session_id="n2", user_input="n8n workflow to back up my notes nightly"
    )

    out = await orch.handle(state)

    # The leading "to" is stripped as a lead-in, leaving the bare description.
    assert n8n.calls == [("back up my notes nightly", False)]
    assert out.response is not None
    lowered = out.response.lower()
    assert "n8n isn't running" in lowered or "docker" in lowered
    assert "confirm" in lowered


async def test_orchestrator_n8n_hook_inert_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orch_mod, "get_settings", lambda: _N8nEnabledSettings(enable_n8n=False)
    )
    n8n = _StubN8nService({"kind": "workflow"})
    # Generous LLM script so however the turn routes (conversation/agent), it
    # completes without exhausting the script — we only assert the hook is inert.
    orch = _orch(n8n, llm_script=["A plain reply, Boss." for _ in range(4)])
    state = GraphState(
        session_id="n3", user_input="Make a workflow on n8n that does a thing."
    )

    await orch.handle(state)

    # Flag off: the n8n service is never reached (the turn routes normally).
    assert n8n.calls == []


class _N8nEnabledSettings:
    """A minimal settings stub exposing only what the n8n hook + persona read.

    The orchestrator reads ``enable_n8n``/``enable_protocols`` (the up-front hook
    guards), ``owner_address`` (persona), ``memory_autowrite`` + ``confirmed``
    handling, and ``enable_self_critique`` through ``get_settings()``; this stub
    exposes exactly those so the monkeypatched call stays cheap and offline.
    """

    def __init__(self, *, enable_n8n: bool) -> None:
        self.enable_n8n = enable_n8n
        self.owner_address = "Boss"
        self.enable_protocols = False
        self.memory_autowrite = False
        self.enable_self_critique = False
