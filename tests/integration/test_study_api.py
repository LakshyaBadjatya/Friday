"""Integration tests for the ``/study`` REST API (Tier 2 study module).

Offline against a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient``
whose ``FRIDAY_ENABLE_STUDY`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the reminders / RAG / studio API tests). No network,
no key.

Covered:
* Every ``/study`` surface is ``404`` when the flag is off.
* ``POST /study/cards`` creates and returns the flashcard.
* ``GET /study/cards?deck=`` lists, filtered by deck.
* ``GET /study/review`` returns the cards due for utcnow.
* ``POST /study/review/{id}`` applies SM-2 and reschedules the card.
* ``DELETE /study/cards/{id}`` removes one.
* ``POST /study/sessions`` + ``GET /study/sessions`` round-trip.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _enable_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated.
    return Settings(
        _env_file=None,
        enable_study=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_study=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose study flag is forced via patched settings."""
    import friday.app as app_module

    factory = _enable_settings if enabled else _disable_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_study_disabled_create_card_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post(
            "/study/cards", json={"deck": "d", "front": "f", "back": "b"}
        )
    assert resp.status_code == 404


def test_study_disabled_list_cards_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/study/cards")
    assert resp.status_code == 404


def test_study_disabled_review_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.get("/study/review")
    assert resp.status_code == 404


def test_study_disabled_review_card_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post("/study/review/1", json={"grade": 4})
    assert resp.status_code == 404


def test_study_disabled_delete_card_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        resp = client.delete("/study/cards/1")
    assert resp.status_code == 404


def test_study_disabled_sessions_are_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_settings()
        post = client.post("/study/sessions", json={"topic": "t", "minutes": 10})
        get = client.get("/study/sessions")
    assert post.status_code == 404
    assert get.status_code == 404


def test_study_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), create is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_settings()
        resp = client.post(
            "/study/cards", json={"deck": "d", "front": "f", "back": "b"}
        )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> full CRUD + review + sessions
# --------------------------------------------------------------------------- #
def test_study_create_and_list_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/study/cards",
            json={"deck": "french", "front": "bonjour", "back": "hello"},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["deck"] == "french"
        assert body["front"] == "bonjour"
        assert body["ease"] == 2.5
        assert body["due_at"] is None
        assert isinstance(body["id"], int)

        client.post(
            "/study/cards",
            json={"deck": "spanish", "front": "hola", "back": "hi"},
        )
        french = client.get("/study/cards?deck=french").json()["cards"]
        assert [c["front"] for c in french] == ["bonjour"]
        all_cards = client.get("/study/cards").json()["cards"]
        assert len(all_cards) == 2


def test_study_create_card_bad_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/study/cards", json={"front": "missing deck"})
    assert resp.status_code == 422


def test_study_review_due_and_advance(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/study/cards",
            json={"deck": "d", "front": "f", "back": "b"},
        )
        cid = created.json()["id"]

        # A brand-new card is due.
        due = client.get("/study/review").json()["cards"]
        assert [c["id"] for c in due] == [cid]

        # Review it with a passing grade -> SM-2 advances + reschedules.
        reviewed = client.post(f"/study/review/{cid}", json={"grade": 4})
        assert reviewed.status_code == 200
        rbody = reviewed.json()
        assert rbody["reps"] == 1
        assert rbody["interval_days"] == 1
        assert rbody["due_at"] is not None

        # No longer due now that it has a future due_at.
        due_after = client.get("/study/review").json()["cards"]
        assert due_after == []


def test_study_review_unknown_card_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        resp = client.post("/study/review/999", json={"grade": 4})
    assert resp.status_code == 404


def test_study_review_bad_grade_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/study/cards", json={"deck": "d", "front": "f", "back": "b"}
        )
        cid = created.json()["id"]
        resp = client.post(f"/study/review/{cid}", json={"grade": 9})
    assert resp.status_code == 422


def test_study_delete_card(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        created = client.post(
            "/study/cards", json={"deck": "d", "front": "f", "back": "b"}
        )
        cid = created.json()["id"]

        deleted = client.delete(f"/study/cards/{cid}")
        assert deleted.status_code == 200
        assert deleted.json()["removed"] == 1
        assert client.get("/study/cards").json()["cards"] == []


def test_study_sessions_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_settings()
        logged = client.post(
            "/study/sessions", json={"topic": "calculus", "minutes": 45}
        )
        assert logged.status_code == 200
        lbody = logged.json()
        assert lbody["topic"] == "calculus"
        assert lbody["minutes"] == 45
        assert lbody["at"]

        client.post("/study/sessions", json={"topic": "history", "minutes": 30})
        listed = client.get("/study/sessions").json()
        # Most-recent first.
        assert [s["topic"] for s in listed["sessions"]] == ["history", "calculus"]
        assert listed["count"] == 2
