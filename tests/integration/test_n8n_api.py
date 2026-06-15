"""Integration tests for the ``/n8n`` REST API (Tier 2 n8n integration).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_N8N`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the protocols / reminders API tests). n8n liveness +
import are ``respx``-mocked; the docker auto-start is a monkeypatched
:func:`asyncio.create_subprocess_exec`. No network, no key leaks.

Covered:
* Every ``/n8n`` surface is ``404`` when the flag is off.
* ``GET /n8n/status`` reports ``up`` (respx-mocked liveness).
* ``POST /n8n/workflow`` when n8n is down + unconfirmed -> ``needs_confirmation``.
* ``POST /n8n/workflow`` when n8n is up + key -> drafted + imported workflow.
* ``POST /n8n/start`` is confirm-gated: unconfirmed -> ``needs_confirmation`` and
  NO subprocess; confirmed -> runs ``start_cmd`` (mocked).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import friday.n8n.service as service_mod
from friday.app import create_app
from friday.config import Settings
from friday.providers.llm import FakeLLM, LLMResponse, Usage

N8N_BASE = "http://localhost:5678"

_VALID_DRAFT = json.dumps(
    {
        "workflow": {
            "name": "Webhook to Slack",
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
        "setup_notes": ["Add Slack credentials"],
    }
)


def _enable_settings(api_key: str | None = "n8n-test-key") -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_n8n=True,
        llm_provider="fake",
        memory_db_path=":memory:",
        n8n_base_url=N8N_BASE,
        n8n_api_key=api_key,
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_n8n=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _install_fake_llm(client: TestClient, *, drafts: int = 1) -> None:
    """Replace the n8n service's drafter LLM with a scripted FakeLLM.

    The app builds the service with a (no-script) FakeLLM by default; we swap in a
    drafter whose LLM returns a valid workflow so the drafting path is exercised
    deterministically.
    """
    service = client.app.state.n8n_service
    service._drafter._llm = FakeLLM(
        responses=[
            LLMResponse(text=_VALID_DRAFT, tool_calls=[], usage=Usage())
            for _ in range(drafts)
        ]
    )


def _client(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool, api_key: str | None = "n8n-test-key"
) -> TestClient:
    """A ``TestClient`` whose n8n flag is forced via patched settings."""
    import friday.app as app_module

    factory = (lambda: _enable_settings(api_key)) if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_n8n_disabled_status_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/n8n/status")
    assert resp.status_code == 404


def test_n8n_disabled_workflow_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/n8n/workflow", json={"description": "x"})
    assert resp.status_code == 404


def test_n8n_disabled_start_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/n8n/start", json={"confirmed": True})
    assert resp.status_code == 404


def test_n8n_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), status is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/n8n/status")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled
# --------------------------------------------------------------------------- #
@respx.mock
def test_n8n_status_reports_up(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(f"{N8N_BASE}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.get("/n8n/status")
    assert resp.status_code == 200
    assert resp.json() == {"up": True}


@respx.mock
def test_n8n_status_reports_down(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(f"{N8N_BASE}/healthz").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{N8N_BASE}/rest/login").mock(side_effect=httpx.ConnectError("down"))
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.get("/n8n/status")
    assert resp.status_code == 200
    assert resp.json() == {"up": False}


@respx.mock
def test_n8n_workflow_down_unconfirmed_needs_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(f"{N8N_BASE}/healthz").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{N8N_BASE}/rest/login").mock(side_effect=httpx.ConnectError("down"))

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no subprocess may run without confirmation")

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", _boom)

    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/n8n/workflow", json={"description": "post to Slack"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "needs_confirmation"
    assert body["action"] == "start_n8n"


@respx.mock
def test_n8n_workflow_up_with_key_drafts_and_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(f"{N8N_BASE}/healthz").mock(return_value=httpx.Response(200))
    import_route = respx.post(f"{N8N_BASE}/api/v1/workflows").mock(
        return_value=httpx.Response(
            200, json={"id": "wf-1", "name": "Webhook to Slack"}
        )
    )

    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        _install_fake_llm(client)
        resp = client.post(
            "/n8n/workflow", json={"description": "webhook to slack"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "workflow"
    assert body["imported"] is True
    assert body["workflow"]["id"] == "wf-1"
    assert body["setup_notes"] == ["Add Slack credentials"]
    # The API key reached n8n only as the X-N8N-API-KEY header.
    assert import_route.calls.last.request.headers["X-N8N-API-KEY"] == "n8n-test-key"


@respx.mock
def test_n8n_workflow_up_without_key_drafts_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(f"{N8N_BASE}/healthz").mock(return_value=httpx.Response(200))
    with _client(monkeypatch, enabled=True, api_key=None) as client:
        client.app.state.settings = _enable_settings(api_key=None)
        _install_fake_llm(client)
        resp = client.post(
            "/n8n/workflow", json={"description": "webhook to slack"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "workflow"
    assert body["imported"] is False
    assert "import_error" not in body
    assert body["workflow"]["name"] == "Webhook to Slack"


def test_n8n_workflow_bad_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/n8n/workflow", json={"nope": "x"})
    assert resp.status_code == 422


def test_n8n_start_unconfirmed_is_needs_confirmation_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no subprocess may run without confirmation")

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", _boom)

    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/n8n/start", json={"confirmed": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "needs_confirmation"
    assert body["action"] == "start_n8n"


def test_n8n_start_confirmed_runs_start_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: dict[str, Any] = {}

    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        spawned["argv"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", _fake_exec)

    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/n8n/start", json={"confirmed": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "start"
    assert body["started"] is True
    # The docker compose argv was passed positionally (no shell), targeting n8n.
    assert spawned["argv"][0] == "docker"
    assert spawned["argv"][-1] == "n8n"
    assert "up" in spawned["argv"]
