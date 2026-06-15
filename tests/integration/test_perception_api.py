"""Integration tests for the ``/perception`` REST API (privacy-heavy, off by default).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_PERCEPTION`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the study / reminders API tests). The perception
service is built from FAKES, so no heavy library (opencv/ultralytics, pytesseract,
pyperclip, mss) is needed, and there is no network, key, or real screen access.

Covered:
* Every ``/perception`` surface is ``404`` when the flag is off.
* ``POST /perception/vision`` returns the fake detections for a base64 image.
* ``POST /perception/ocr`` returns the fake OCR text for a base64 image.
* ``GET`` / ``POST /perception/clipboard`` round-trip text.
* ``POST /perception/screen`` returns ``{ocr_text, detections}``.
* A bad/non-base64 image body is ``422``.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_perception=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_perception=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose perception flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


def _b64(payload: bytes = b"fake-image-bytes") -> str:
    return base64.b64encode(payload).decode("ascii")


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_vision_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/perception/vision", json={"image_b64": _b64()})
    assert resp.status_code == 404


def test_ocr_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/perception/ocr", json={"image_b64": _b64()})
    assert resp.status_code == 404


def test_clipboard_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        get = client.get("/perception/clipboard")
        post = client.post("/perception/clipboard", json={"text": "x"})
    assert get.status_code == 404
    assert post.status_code == 404


def test_screen_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/perception/screen")
    assert resp.status_code == 404


def test_perception_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), vision is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/perception/vision", json={"image_b64": _b64()})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> each route works with fakes
# --------------------------------------------------------------------------- #
def test_vision_returns_fake_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/perception/vision", json={"image_b64": _b64()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["detections"][0]["label"] == "person"
    assert body["detections"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_ocr_returns_fake_text(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/perception/ocr", json={"image_b64": _b64()})
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello world"


def test_clipboard_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        # Fresh fake clipboard starts empty.
        assert client.get("/perception/clipboard").json()["text"] == ""
        wrote = client.post("/perception/clipboard", json={"text": "copied!"})
        assert wrote.status_code == 200
        assert wrote.json()["ok"] is True
        assert client.get("/perception/clipboard").json()["text"] == "copied!"


def test_screen_describe_returns_ocr_and_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/perception/screen")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ocr_text"] == "hello world"
    assert body["detections"][0]["label"] == "person"


def test_vision_bad_base64_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/perception/vision", json={"image_b64": "not base64!!!"})
    assert resp.status_code == 422


def test_ocr_missing_field_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/perception/ocr", json={})
    assert resp.status_code == 422
