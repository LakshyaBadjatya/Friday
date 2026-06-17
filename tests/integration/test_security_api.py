# © Lakshya Badjatya — Author
"""Integration tests for the /security rotation + audit-anchor surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, **flags: str) -> TestClient:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return TestClient(create_app())


def test_rotation_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post(
            "/security/rotation", json={"secrets": [], "max_age_seconds": 100}
        )
    assert resp.status_code == 404


def test_rotation_disabled_404s_even_for_invalid_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The flag check must precede body validation: a disabled feature returns 404
    # even for a malformed body (a bound param would 422 first, leaking existence).
    with _client(monkeypatch) as client:
        resp = client.post(
            "/security/rotation", json={"secrets": [], "max_age_seconds": -1}
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "secret rotation disabled"


def test_rotation_enabled_still_422s_for_invalid_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When enabled, validation is preserved: an invalid body is a 422.
    with _client(monkeypatch, FRIDAY_ENABLE_SECRET_ROTATION="true") as client:
        resp = client.post(
            "/security/rotation", json={"secrets": [], "max_age_seconds": -1}
        )
    assert resp.status_code == 422


def test_rotation_reports_due(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "secrets": [
            {"name": "old", "last_rotated_ts": 0},
            {"name": "fresh", "last_rotated_ts": 9999999999},
        ],
        "max_age_seconds": 100,
    }
    with _client(monkeypatch, FRIDAY_ENABLE_SECRET_ROTATION="true") as client:
        resp = client.post("/security/rotation", json=body)
    assert resp.status_code == 200
    assert resp.json()["due"] == ["old"]  # epoch-0 secret is overdue; future one isn't


def test_anchor_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post("/security/anchor")
    assert resp.status_code == 404


def test_anchor_pins_head_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor_file = tmp_path / "anchors.jsonl"
    client = _client(
        monkeypatch,
        FRIDAY_ENABLE_AUDIT_ANCHOR="true",
        FRIDAY_AUDIT_ANCHOR_PATH=str(anchor_file),
        FRIDAY_AUDIT_LEDGER_PATH=str(tmp_path / "audit.jsonl"),  # empty ledger
    )
    with client:
        resp = client.post("/security/anchor")
    assert resp.status_code == 200
    body = resp.json()
    assert body["head_hash"] == "0" * 64  # GENESIS head for an empty ledger
    assert anchor_file.exists() and anchor_file.read_text().strip()  # one anchor line
