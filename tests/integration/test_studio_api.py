"""Integration tests for the ``/studio`` API (Phase 7, Stage 1).

All offline against the :class:`~friday.providers.llm.FakeLLM` wired onto
``app.state.studio`` after the lifespan, plus a ``TestClient`` with the studio
flag forced on/off. No network, no key.

Covered:
* ``POST /studio/generate`` enabled -> a valid procedural scene (``kind=="scene"``)
  whose ``scene`` round-trips through :func:`~friday.studio.scene.validate_scene`.
* The same route disabled (flag off) -> ``404`` (the feature does not exist).
* Hi-fi requested with no key -> falls back to procedural (``kind=="scene"``),
  never paywalls / errors.
* ``description`` validation: an empty description -> ``422``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.studio.generator import ProceduralGenerator, StudioService
from friday.studio.scene import validate_scene

_VALID_SCENE_JSON = json.dumps(
    {
        "name": "red-cube",
        "background": "#101014",
        "nodes": [
            {
                "id": "cube",
                "type": "box",
                "params": {"w": 1.0, "h": 1.0, "d": 1.0},
                "color": "#ff0000",
            }
        ],
    }
)


def _enable_studio_settings() -> Settings:
    return Settings(_env_file=None, enable_studio=True, llm_provider="fake")


def _disable_studio_settings() -> Settings:
    return Settings(_env_file=None, enable_studio=False, llm_provider="fake")


def _scripted_service(*texts: str, hifi: object | None = None) -> StudioService:
    """A :class:`StudioService` whose procedural path is a scripted FakeLLM."""
    llm = FakeLLM(
        responses=[LLMResponse(text=t, tool_calls=[], usage=Usage()) for t in texts]
    )
    return StudioService(ProceduralGenerator(llm), hifi=hifi)  # type: ignore[arg-type]


def test_post_studio_generate_disabled_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the studio flag is off, ``POST /studio/generate`` is 404."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _disable_studio_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _disable_studio_settings()
        resp = client.post("/studio/generate", json={"description": "a red cube"})

    assert resp.status_code == 404


def test_get_studio_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The studio index page is also absent when the flag is off."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _disable_studio_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _disable_studio_settings()
        resp = client.get("/studio")

    assert resp.status_code == 404


def test_post_studio_generate_enabled_returns_valid_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled fast generation returns ``kind=='scene'`` with a valid Scene body."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _enable_studio_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _enable_studio_settings()
        app.state.studio = _scripted_service(_VALID_SCENE_JSON)
        resp = client.post("/studio/generate", json={"description": "a red cube"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "scene"
    # The returned scene round-trips through the shared contract validator.
    scene = validate_scene(body["scene"])
    assert scene.name == "red-cube"
    assert scene.nodes[0].type == "box"


def test_post_studio_generate_hifi_no_key_falls_back_to_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hi-fi requested with no provider wired falls back to a procedural scene."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _enable_studio_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _enable_studio_settings()
        # hifi=None models a keyless deployment: must fall back, never paywall.
        app.state.studio = _scripted_service(_VALID_SCENE_JSON, hifi=None)
        resp = client.post(
            "/studio/generate",
            json={"description": "a red cube", "quality": "hifi"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "scene"
    assert body["scene"]["name"] == "red-cube"


def test_post_studio_generate_rejects_empty_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``description`` is rejected with 422 (min_length=1)."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _enable_studio_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _enable_studio_settings()
        app.state.studio = _scripted_service(_VALID_SCENE_JSON)
        resp = client.post("/studio/generate", json={"description": ""})

    assert resp.status_code == 422


def test_studio_default_off_post_is_404() -> None:
    """With pristine env-default settings (flag off), the route is 404."""
    # Uses the real default settings path: a fresh app with the flag at its
    # default False must not expose the studio route.
    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _disable_studio_settings()
        resp = client.post("/studio/generate", json={"description": "x"})
    assert resp.status_code == 404
