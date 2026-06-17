"""POST /emotion/enroll — owner personalization enrollment (Phase 3)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.api.routes_emotion import router
from friday.providers.emotion import FakeEmotion
from friday.voice.fixtures import make_wav


def _client(tmp_path, enabled=True, base=None):
    app = FastAPI()
    app.include_router(router)
    app.state.settings = SimpleNamespace(
        enable_emotion=enabled, emotion_calibration=str(tmp_path / "cal.json"),
    )
    app.state.emotion_base_provider = base
    return TestClient(app), tmp_path / "cal.json"


def test_enroll_writes_calibration_and_recenters(tmp_path) -> None:
    # Owner's neutral reads as (0.3, 0.6, 0.5) -> offset (0.2, -0.1, 0.0).
    client, cal_path = _client(tmp_path, base=FakeEmotion(valence=0.3, arousal=0.6, dominance=0.5))
    wav_b64 = base64.b64encode(make_wav(seconds=0.3)).decode()
    r = client.post("/emotion/enroll", json={"clips": [wav_b64]})
    assert r.status_code == 200
    body = r.json()
    assert abs(body["calibration"]["v_off"] - 0.2) < 1e-6
    assert abs(body["calibration"]["a_off"] + 0.1) < 1e-6
    assert body["clips"] == 1
    assert cal_path.is_file()


def test_enroll_404_when_disabled(tmp_path) -> None:
    client, _ = _client(tmp_path, enabled=False, base=FakeEmotion())
    wav_b64 = base64.b64encode(make_wav(seconds=0.2)).decode()
    r = client.post("/emotion/enroll", json={"clips": [wav_b64]})
    assert r.status_code == 404
