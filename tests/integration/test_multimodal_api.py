# © Lakshya Badjatya — Author
"""Integration tests for /imagegen + /pdf/layout (Wave C; default off)."""

from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_multimodal as routes_multimodal
from friday.api.routes_multimodal import router as multimodal_router
from friday.config import Settings


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(multimodal_router)
    return app


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_imagegen_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_multimodal, "get_settings", _settings)
    with TestClient(_app()) as client:
        resp = client.post("/imagegen", json={"prompt": "an owl"})
    assert resp.status_code == 404


def test_imagegen_enabled_returns_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes_multimodal, "get_settings", lambda: _settings(enable_imagegen=True)
    )
    with TestClient(_app()) as client:
        resp = client.post("/imagegen", json={"prompt": "an owl in moonlight"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt"] == "an owl in moonlight"
    assert body["media_type"] in ("image/svg+xml", "image/png")
    assert body["data_base64"]


def test_pdf_layout_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_multimodal, "get_settings", _settings)
    payload = base64.b64encode(b"hello").decode("ascii")
    with TestClient(_app()) as client:
        resp = client.post("/pdf/layout", json={"pdf_base64": payload})
    assert resp.status_code == 404


def test_pdf_layout_enabled_extracts_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes_multimodal, "get_settings", lambda: _settings(enable_pdf_layout=True)
    )
    payload = base64.b64encode(b"Heading\n\nBody paragraph.").decode("ascii")
    with TestClient(_app()) as client:
        resp = client.post("/pdf/layout", json={"pdf_base64": payload})
    assert resp.status_code == 200
    pages = resp.json()["pages"]
    assert pages[0]["blocks"] == ["Heading", "Body paragraph."]


def test_pdf_layout_rejects_bad_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes_multimodal, "get_settings", lambda: _settings(enable_pdf_layout=True)
    )
    with TestClient(_app()) as client:
        resp = client.post("/pdf/layout", json={"pdf_base64": "!!!not base64!!!"})
    assert resp.status_code == 400
    assert resp.json()["type"] == "BadInput"
